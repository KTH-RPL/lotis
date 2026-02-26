"""
LoTIS: Learning to Localize Trajectories in Image Space

A library for localizing query images within recorded trajectories.
"""

from .localizer import TrajectoryLocalizer, TrajectoryEncoding, BatchedTrajectoryEncoding
from .result import LocalizationResult

__version__ = "0.1.0"
__all__ = [
    "TrajectoryLocalizer",
    "TrajectoryEncoding",
    "BatchedTrajectoryEncoding",
    "LocalizationResult",
]
