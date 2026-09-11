"""Reusable interactive environments for experimental agent loops.

Environment managers own the lifecycle of interactive environment resources for
custom ``AgentLoopBase`` implementations.
"""

from .base import EnvironmentManagerBase, EnvironmentReset, EnvironmentStep
from .alfworld import ALFWorldEnvironmentManager

__all__ = [
    "ALFWorldEnvironmentManager",
    "EnvironmentManagerBase",
    "EnvironmentReset",
    "EnvironmentStep",
]
