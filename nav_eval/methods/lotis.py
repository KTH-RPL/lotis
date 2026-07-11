"""LoTIS method adapter for nav_eval."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from lotis import TrajectoryLocalizer

from ..interfaces import Action, NavigationMethod
from ..registry import register_method


class LoTISNavigationMethod(NavigationMethod):
    """Navigation policy driven by released LoTIS localization predictions."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.localizer: Optional[TrajectoryLocalizer] = None
        self.encoding = None
        self.tracking_mode = "forward"
        self.goal_point_id = None
        self.simulator = None
        self._last_result = None
        self._traj_frames = None

        self.visibility_threshold = config.get("visibility_threshold", 0.5)
        self.forward_velocity = config.get("forward_velocity", 0.4)
        self.turn_gain = config.get("turn_gain", 2.0)
        self.use_geodesic_follower = config.get("use_geodesic_follower", True)
        self.geodesic_forward_distance = config.get("geodesic_forward_distance", 1.0)

    def setup(self) -> None:
        print("  Loading LoTIS localizer...")
        self.localizer = TrajectoryLocalizer.from_checkpoint(
            checkpoint_path=self.config.get("checkpoint_path", "final_model.pth"),
            config_path=self.config.get("config_path", "final_config.yaml"),
            device=self.config.get("device", "cuda"),
            dinov3_weights=self.config.get("dinov3_weights"),
            dinov3_repo=self.config.get("dinov3_repo", "./dinov3"),
        )
        print("  Setup complete!")

    def set_simulator(self, simulator: Any) -> None:
        self.simulator = simulator

    def reset_episode(self, episode: Dict[str, Any]) -> None:
        if self.localizer is None:
            raise RuntimeError("Method is not set up")
        self.tracking_mode = episode["tracking_mode"]
        self.goal_point_id = episode["goal_point_id"]
        self._traj_frames = episode["rgb_images"]
        self._last_result = None
        self.encoding = self.localizer.encode_trajectory(episode["rgb_images"])

    def act(self, observation: Dict[str, Any]) -> Action:
        if self.localizer is None or self.encoding is None:
            raise RuntimeError("No active episode")

        result = self.localizer.localize(observation["rgb_pil"], self.encoding)
        self._last_result = result
        coords = result.coords
        visibility = result.visibility
        distances = result.distances
        if distances is None:
            raise RuntimeError("LoTIS evaluation requires distance predictions")

        visible_indices = np.where(visibility > self.visibility_threshold)[0]
        is_recovering = False
        if len(visible_indices) == 0:
            visible_indices = np.arange(len(coords))
            is_recovering = False

        visible_dists = distances[visible_indices]
        visible_coords = coords[visible_indices]
        if self.tracking_mode == "backward":
            visible_dists = visible_dists[::-1]
            visible_coords = visible_coords[::-1]

        min_dist_id = int(np.argmin(visible_dists))
        min_dist_id_global = int(visible_indices[min_dist_id])

        if self.tracking_mode == "goal_point":
            is_forward = min_dist_id_global <= self.goal_point_id
            goal_local_ids = np.where(visible_indices == self.goal_point_id)[0]
            max_valid = len(visible_coords) - 1
            min_valid = 0
            if len(goal_local_ids) > 0:
                max_valid = int(goal_local_ids[0])
                min_valid = int(goal_local_ids[0])
            if is_forward:
                goal_offset = max_valid if is_recovering else min(min_dist_id + 1, max_valid)
            else:
                goal_offset = min_valid if is_recovering else max(min_dist_id - 1, min_valid)
        else:
            goal_offset = len(visible_coords) - 1 if is_recovering else min(min_dist_id + 1, len(visible_coords) - 1)

        goal_coord = visible_coords[goal_offset]
        x_normalized = goal_coord[1]
        intrinsics = observation["intrinsics"]
        fx = intrinsics[0, 0]
        cx = intrinsics[0, 2]
        width = cx * 2.0
        pixel_offset = x_normalized * (width / 2.0)
        angle_to_goal = np.arctan(pixel_offset / fx)

        if self.use_geodesic_follower and self.simulator is not None:
            return self._geodesic_follower_action(observation, angle_to_goal)

        angular_velocity = np.clip(self.turn_gain * angle_to_goal, -1.0, 1.0)
        return Action(self.forward_velocity, -float(angular_velocity))

    def _geodesic_follower_action(self, observation: Dict[str, Any], angle_to_goal: float) -> Action:
        import habitat_sim

        current_pos = observation["position"]
        current_rot = observation["rotation"]
        rot_coeffs = habitat_sim.utils.common.quat_to_coeffs(current_rot)
        x, y, z, w = rot_coeffs
        siny_cosp = 2 * (w * y + x * z)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        current_yaw = np.arctan2(siny_cosp, cosy_cosp)

        target_yaw = current_yaw - angle_to_goal
        forward_x = -np.sin(target_yaw)
        forward_z = -np.cos(target_yaw)
        goal_pos = np.array(
            [
                current_pos[0] + forward_x * self.geodesic_forward_distance,
                current_pos[1],
                current_pos[2] + forward_z * self.geodesic_forward_distance,
            ],
            dtype=np.float32,
        )

        pathfinder = self.simulator.sim.pathfinder
        if pathfinder.is_navigable(goal_pos):
            snapped_goal = goal_pos
        else:
            snapped_goal = pathfinder.snap_point(goal_pos)
            if not np.isfinite(snapped_goal).all():
                angular_velocity = np.clip(self.turn_gain * angle_to_goal, -1.0, 1.0)
                return Action(self.forward_velocity, -float(angular_velocity))

        path_points, _ = self.simulator.compute_path(current_pos, snapped_goal)
        if path_points is None or len(path_points) < 2:
            angular_velocity = np.clip(self.turn_gain * angle_to_goal, -1.0, 1.0)
            return Action(self.forward_velocity, -float(angular_velocity))

        waypoint = np.asarray(path_points[1])
        dx = waypoint[0] - current_pos[0]
        dz = waypoint[2] - current_pos[2]
        waypoint_yaw = np.arctan2(-dx, -dz)
        angular_error = waypoint_yaw - current_yaw
        angular_error = np.arctan2(np.sin(angular_error), np.cos(angular_error))
        angular_velocity = np.clip(self.turn_gain * angular_error, -1.0, 1.0)
        turn_factor = 1.0 - min(abs(angular_error) / np.pi, 0.8)
        linear_velocity = self.forward_velocity * turn_factor
        return Action(float(linear_velocity), float(angular_velocity))

    def visualize(self, observation: Dict[str, Any]) -> None:
        import cv2
        import rerun as rr

        result = self._last_result
        if result is None:
            return

        rgb = observation["rgb"]
        h, w = rgb.shape[:2]

        vis_mask = result.visibility > self.visibility_threshold
        vis_indices = np.where(vis_mask)[0]
        if vis_indices.size == 0:
            rr.log("camera/points", rr.Clear(recursive=False))
            rr.log("camera/lines", rr.Clear(recursive=False))
            return

        vis_coords = result.coords[vis_mask]  # (row, col) in [-1, 1]
        px = (vis_coords[:, 1] + 1.0) / 2.0 * w
        py = (vis_coords[:, 0] + 1.0) / 2.0 * h
        positions = np.stack([px, py], axis=-1)

        # Colour points by trajectory index: blue = start, red = end.
        n_traj = len(result.visibility)
        cvals = (255 * vis_indices / max(1, n_traj - 1)).astype(np.uint8)
        colors_rgb = cv2.applyColorMap(cvals[:, None], cv2.COLORMAP_JET)[:, 0, ::-1]

        # Larger points are closer to the camera.
        base_r = max(h, w) * 0.015
        if result.distances is not None:
            d = result.distances[vis_mask]
            d_min, d_max = d.min(), d.max()
            norm = (d - d_min) / (d_max - d_min) if d_max > d_min else np.zeros_like(d)
            radii = base_r * (1.0 - 0.7 * norm)
        else:
            radii = np.full(len(vis_indices), base_r)

        rr.log("camera/points", rr.Points2D(positions=positions, colors=colors_rgb, radii=radii))

        if len(positions) > 1:
            consecutive = vis_indices[1:] == vis_indices[:-1] + 1
            segments = np.stack([positions[:-1], positions[1:]], axis=1)[consecutive]
            seg_colors = colors_rgb[:-1][consecutive]
            rr.log("camera/lines", rr.LineStrips2D(segments, colors=seg_colors))

        if self._traj_frames is not None:
            closest = result.closest_frame()
            frame = np.asarray(self._traj_frames[closest])
            rr.log("closest_trajectory_frame", rr.Image(frame).compress(jpeg_quality=90))

    def cleanup(self) -> None:
        self.localizer = None
        self.encoding = None


register_method("lotis", LoTISNavigationMethod)
