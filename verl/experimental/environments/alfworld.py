"""ALFWorld text-environment manager for verl AgentLoop implementations.

The manager runs exactly one ``game.tw-pddl`` task at a time.  It intentionally
does not use ``AlfredTWEnv.collect_game_files()``: training data already names
the exact task in ``extra_info.game_file``, so scanning an entire ALFWorld split
for every rollout would be both unnecessary and slow.
"""

from __future__ import annotations

import os
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, Mapping
from uuid import uuid4

from .base import EnvironmentManagerBase, EnvironmentReset, EnvironmentStep


def _first(value: Any) -> Any:
    """Extract the only item returned by TextWorld's batch-size-one API."""
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"Expected one ALFWorld environment result, got {len(value)}.")
        return value[0]
    return value


def _unbatch_info(info: Mapping[str, Any]) -> dict[str, Any]:
    return {key: _first(value) for key, value in info.items()}


class _ALFWorldEnvironmentSlot:
    """One long-lived, non-concurrent TextWorld/PddlEnv instance."""

    def __init__(
        self,
        *,
        max_episode_steps: int = 50,
        domain_randomization: bool = False,
    ) -> None:
        if max_episode_steps <= 0:
            raise ValueError(f"max_episode_steps must be positive, got {max_episode_steps}.")

        self.max_episode_steps = max_episode_steps
        self.domain_randomization = domain_randomization
        self._env: Any | None = None
        self._env_id: str | None = None
        self._game_file: Path | None = None
        self._started = False

    def reset(self, game_file: Path) -> EnvironmentReset:
        """Load a game into this slot and start its episode.

        TextWorld retains the underlying ``PddlEnv`` when a batch environment
        is reset and then loaded with another PDDL game.  Reusing that object
        is intentional: every fresh ``PddlEnv`` loads a separate Fast Downward
        shared library, whose temporary backing file cannot be safely unloaded
        during a long-running worker process.
        """
        if self._env is None:
            self._create_env(game_file)
        elif game_file != self._game_file:
            # TextworldBatchGymEnv chooses a game from this list during reset.
            # Reseeding rebuilds its internal shuffled iterator for the new,
            # single-game list without constructing a fresh PddlEnv.
            self._env.gamefiles = [str(game_file)]
            self._env.seed(1234)

        observation, info = self._env.reset()
        self._game_file = game_file
        self._started = True
        return EnvironmentReset(observation=str(_first(observation)), info=_unbatch_info(info))

    def step(self, action: str) -> EnvironmentStep:
        """Execute one textual ALFWorld action."""
        if not self._started or self._env is None:
            raise RuntimeError("Call reset() before step().")
        if not isinstance(action, str) or not action.strip():
            raise ValueError("ALFWorld action must be a non-empty string.")

        result = self._env.step([action.strip()])
        if len(result) != 4:
            raise RuntimeError(f"Unexpected ALFWorld step result with {len(result)} values.")
        observation, reward, done, info = result
        done = bool(_first(done))
        if done:
            self._started = False
        return EnvironmentStep(
            observation=str(_first(observation)),
            reward=float(_first(reward)),
            done=done,
            info=_unbatch_info(info),
        )

    def close(self) -> None:
        """Close the TextWorld wrapper without forcing ``dlclose``.

        ``fast_downward.close_lib`` is intentionally not called.  It can
        invalidate C-level state still referenced by TextWorld; the worker
        process owns final OS-level cleanup when it exits.
        """
        if self._env is not None:
            self._env.close()
        if self._env_id is not None:
            # TextWorld keeps dynamically registered environments in a module
            # global registry.  Removing the entry prevents unbounded growth
            # when a long-running worker handles many distinct games.
            import textworld.gym

            textworld.gym.registry.pop(self._env_id, None)

        self._env = None
        self._env_id = None
        self._game_file = None
        self._started = False

    def _create_env(self, game_file: Path) -> None:
        try:
            import textworld
            import textworld.gym
            from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos
        except ImportError as exc:
            raise ImportError(
                "ALFWorld text environments require the optional 'alfworld' and 'textworld' packages."
            ) from exc

        request_infos = textworld.EnvInfos(won=True)
        wrappers = [AlfredDemangler(shuffle=self.domain_randomization), AlfredInfos]
        self._env_id = textworld.gym.register_games(
            [str(game_file)],
            request_infos=request_infos,
            batch_size=1,
            auto_reset=False,
            max_episode_steps=self.max_episode_steps,
            asynchronous=False,
            name=f"verl-alfworld-{uuid4().hex}",
            wrappers=wrappers,
        )
        self._env = textworld.gym.make(self._env_id)
        self._game_file = game_file


class ALFWorldEnvironmentLease:
    """A non-shareable ALFWorld environment slot checked out by one AgentLoop."""

    def __init__(
        self,
        manager: "ALFWorldEnvironmentManager",
        slot: _ALFWorldEnvironmentSlot,
        reset: EnvironmentReset,
    ) -> None:
        self._manager = manager
        self._slot = slot
        self.reset = reset

    async def step(self, action: str) -> EnvironmentStep:
        """Execute one transition through the worker's TextWorld critical section."""
        return await self._manager._step(self._slot, action)


class ALFWorldEnvironmentManager(EnvironmentManagerBase):
    """Worker-owned pool of reusable ALFWorld text-environment slots.

    The manager is created once in an ``AgentLoopWorker``.  An AgentLoop leases
    one slot for the entire trajectory, so its PDDL state is isolated from
    other trajectories.  Once released, that slot loads the next game without
    constructing another ``PddlEnv`` or another Fast Downward shared library.
    """

    def __init__(self, *, num_slots: int = 16, max_episode_steps: int = 50, domain_randomization: bool = False):
        if num_slots <= 0:
            raise ValueError(f"num_slots must be positive, got {num_slots}.")
        self.num_slots = num_slots
        self._closed = False
        self._available: asyncio.Queue[_ALFWorldEnvironmentSlot] = asyncio.Queue(maxsize=num_slots)
        # TextWorld's TaTsu grammar parser is process-global and is not thread
        # safe. Both load/reset and step derive grammar text through it, so all
        # TextWorld calls must be serialized within one Ray worker process.
        self._textworld_lock = asyncio.Lock()
        self._slots = [
            _ALFWorldEnvironmentSlot(
                max_episode_steps=max_episode_steps,
                domain_randomization=domain_randomization,
            )
            for _ in range(num_slots)
        ]
        for slot in self._slots:
            self._available.put_nowait(slot)

    @staticmethod
    def resolve_game_file(task: Mapping[str, Any]) -> Path:
        """Resolve the relative game path stored in an ALFWorld dataset row."""
        game_file = task.get("game_file")
        if not game_file:
            raise KeyError("ALFWorld task metadata must contain 'game_file'.")

        path = Path(str(game_file)).expanduser()
        if not path.is_absolute():
            root_env = str(task.get("data_root_env", "ALFWORLD_DATA"))
            data_root = os.environ.get(root_env)
            if not data_root:
                raise EnvironmentError(f"Environment variable {root_env!r} is required to resolve {path}.")
            path = Path(data_root).expanduser() / path

        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"ALFWorld game file does not exist: {path}")
        return path

    @asynccontextmanager
    async def episode(self, task: Mapping[str, Any]) -> AsyncGenerator[ALFWorldEnvironmentLease, None]:
        """Lease a slot, reset it to ``task``, and return it when the loop ends."""
        if self._closed:
            raise RuntimeError("Cannot lease a closed ALFWorldEnvironmentManager.")
        game_file = self.resolve_game_file(task)
        slot = await self._available.get()
        try:
            async with self._textworld_lock:
                reset = await asyncio.to_thread(slot.reset, game_file)
            yield ALFWorldEnvironmentLease(self, slot, reset)
        finally:
            self._available.put_nowait(slot)

    async def _step(self, slot: _ALFWorldEnvironmentSlot, action: str) -> EnvironmentStep:
        async with self._textworld_lock:
            return await asyncio.to_thread(slot.step, action)

    def close(self) -> None:
        """Close all slots when the owning AgentLoopWorker is being torn down."""
        if self._closed:
            return
        for slot in self._slots:
            slot.close()
        self._closed = True
