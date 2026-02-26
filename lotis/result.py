"""Result dataclass for localization predictions."""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class LocalizationResult:
    """
    Result of localizing a query image within a trajectory.

    Attributes:
        coords: Array of shape [N, 2] with (row, col) coordinates in [-1, 1] range
                for each trajectory frame. Row corresponds to vertical position,
                col to horizontal position.
        visibility: Array of shape [N] with visibility probabilities [0, 1] for each frame.
                   Higher values indicate the query location is visible in that frame.
        distances: Optional array of shape [N] with relative distances to each frame.
                  Lower values indicate the query is closer to that trajectory position.
        num_frames: Number of frames in the trajectory.
    """
    coords: np.ndarray
    visibility: np.ndarray
    distances: Optional[np.ndarray] = None
    num_frames: int = 0

    def visible_coords(self, threshold: float = 0.5) -> np.ndarray:
        """
        Get coordinates only for frames where visibility exceeds threshold.

        Args:
            threshold: Minimum visibility probability to include a frame.

        Returns:
            Array of shape [M, 2] with coordinates of visible frames.
        """
        mask = self.visibility > threshold
        return self.coords[mask]

    def visible_indices(self, threshold: float = 0.5) -> np.ndarray:
        """
        Get frame indices where visibility exceeds threshold.

        Args:
            threshold: Minimum visibility probability.

        Returns:
            Array of frame indices.
        """
        return np.where(self.visibility > threshold)[0]

    def closest_frame(self) -> int:
        """
        Get the index of the frame closest to the query (if distances available).
        Falls back to frame with highest visibility if distances not available.

        Returns:
            Frame index.
        """
        if self.distances is not None:
            visible_mask = self.visibility > 0.5
            if visible_mask.any():
                masked_distances = np.where(visible_mask, self.distances, np.inf)
                return int(np.argmin(masked_distances))
        return int(np.argmax(self.visibility))

    def to_pixel_coords(self, image_height: int, image_width: int) -> np.ndarray:
        """
        Convert normalized coordinates to pixel coordinates.

        Args:
            image_height: Height of the query image in pixels.
            image_width: Width of the query image in pixels.

        Returns:
            Array of shape [N, 2] with (y, x) pixel coordinates.
        """
        pixel_coords = (self.coords + 1.0) / 2.0 * np.array([image_height, image_width])
        return pixel_coords.astype(np.int32)
