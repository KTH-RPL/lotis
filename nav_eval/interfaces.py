"""Public method interface for navigation evaluation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class Action:
    """Velocity command returned by an evaluation method."""

    linear_velocity: float
    angular_velocity: float


class NavigationMethod(ABC):
    """Base class for methods that can run in the evaluation."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def setup(self) -> None:
        """Load models and initialize persistent state."""

    @abstractmethod
    def reset_episode(self, episode: Dict[str, Any]) -> None:
        """Reset method state for a new trajectory-query episode."""

    @abstractmethod
    def act(self, observation: Dict[str, Any]) -> Action:
        """Return a velocity command for the current observation."""

    def set_simulator(self, simulator: Any) -> None:
        """Receive the active simulator wrapper if the method needs pathfinding."""

    def visualize(self, observation: Dict[str, Any]) -> None:
        """Log method-specific visualizations to Rerun.

        Called each step (after ``act``) only when the runner is started with
        visualization enabled. The runner already logs the live camera frame at
        entity path ``"camera"`` and the goal image at ``"goal"``; log overlays
        under ``"camera/..."`` so they compose with the live view. No-op by default.
        """

    def cleanup(self) -> None:
        """Release resources after the evaluation run."""
