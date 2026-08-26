"""Reusable interactive environments for experimental agent loops.

Environment managers own the lifecycle of one environment instance.  They are
deliberately independent of the rollout backend so they can be used by custom
``AgentLoopBase`` implementations without coupling an environment to Ray,
vLLM, or SGLang.
"""

from .base import EnvironmentManagerBase, EnvironmentReset, EnvironmentStep
from .alfworld import ALFWorldEnvironmentLease, ALFWorldEnvironmentManager

__all__ = [
    "ALFWorldEnvironmentManager",
    "ALFWorldEnvironmentLease",
    "EnvironmentManagerBase",
    "EnvironmentReset",
    "EnvironmentStep",
]
