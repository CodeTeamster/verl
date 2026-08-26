"""Common interfaces for environments used by experimental agent loops."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class EnvironmentReset:
    """Result returned when an environment starts a new episode."""

    observation: str
    info: dict[str, Any]


@dataclass(slots=True)
class EnvironmentStep:
    """Result of one environment transition."""

    observation: str
    reward: float
    done: bool
    info: dict[str, Any]


class EnvironmentManagerBase(ABC):
    """Base class for a worker-owned environment manager or environment pool.

    A concrete manager may expose one environment directly, or lease multiple
    isolated environment slots to concurrently running agent trajectories.
    """

    @abstractmethod
    def close(self) -> None:
        """Release all local resources owned by this manager."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
