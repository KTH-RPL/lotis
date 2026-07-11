"""Loader for pre-generated navigation evaluation datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import habitat_sim
import numpy as np
from PIL import Image


class NavigationDataset:
    """Iterates over trajectory-query pairs from an existing evaluation dataset."""

    def __init__(
        self,
        dataset_path: str,
        scenes_to_eval: Optional[List[str]] = None,
        max_trajectories_per_scene: Optional[int] = None,
        max_episodes: Optional[int] = None,
        max_queries_per_trajectory: Optional[int] = None,
        use_augmented: bool = True,
        query_type: str = "off_traj",
        tracking_mode: str = "forward",
    ):
        self.dataset_path = Path(dataset_path)
        self.trajectories_path = self.dataset_path / "trajectories"
        self.scenes_to_eval = scenes_to_eval
        self.max_trajectories_per_scene = max_trajectories_per_scene
        self.max_episodes = max_episodes
        self.max_queries_per_trajectory = max_queries_per_trajectory
        self.use_augmented = use_augmented

        if tracking_mode not in {"forward", "backward", "goal_point"}:
            raise ValueError("tracking_mode must be 'forward', 'backward', or 'goal_point'")
        if tracking_mode == "goal_point" and query_type != "off_traj":
            raise ValueError("goal_point tracking mode is only supported for off_traj queries")
        self.tracking_mode = tracking_mode

        if query_type == "off_traj":
            self.query_type = "off_traj"
        elif query_type == "on_traj":
            self.query_type = "on_traj_backward" if tracking_mode == "backward" else "on_traj_forward"
        else:
            raise ValueError("query_type must be 'off_traj' or 'on_traj'")

        if not self.trajectories_path.exists():
            raise ValueError(f"Trajectories path not found: {self.trajectories_path}")

        self.episode_infos = self._discover_episodes()
        if self.max_episodes is not None:
            self.episode_infos = self.episode_infos[: self.max_episodes]

        unique = {(e["scene_id"], e["trajectory_id"]) for e in self.episode_infos}
        scenes = {e["scene_id"] for e in self.episode_infos}
        print(
            f"Found {len(self.episode_infos)} trajectory-query pairs "
            f"({len(unique)} unique trajectories) across {len(scenes)} scenes"
        )

    def _discover_episodes(self) -> List[Dict[str, str]]:
        episodes = []

        for scene_dir in sorted(self.trajectories_path.iterdir()):
            if not scene_dir.is_dir():
                continue
            scene_id = scene_dir.name
            if self.scenes_to_eval is not None and scene_id not in self.scenes_to_eval:
                continue

            traj_count = 0
            for traj_dir in sorted(scene_dir.iterdir()):
                if not traj_dir.is_dir():
                    continue
                metadata_path = traj_dir / "trajectory_metadata.json"
                queries_dir = traj_dir / "queries"
                if not metadata_path.exists() or not queries_dir.exists():
                    continue

                query_dirs = sorted(d for d in queries_dir.iterdir() if d.is_dir())
                query_dirs = [d for d in query_dirs if d.name.startswith(self.query_type)]

                if self.max_queries_per_trajectory is not None:
                    if self.tracking_mode == "backward":
                        query_dirs = query_dirs[-self.max_queries_per_trajectory :]
                    else:
                        query_dirs = query_dirs[: self.max_queries_per_trajectory]

                for query_dir in query_dirs:
                    episodes.append(
                        {
                            "scene_id": scene_id,
                            "trajectory_id": traj_dir.name,
                            "path": str(traj_dir),
                            "metadata_path": str(metadata_path),
                            "query_dir": str(query_dir),
                            "query_id": query_dir.name,
                        }
                    )

                traj_count += 1
                if (
                    self.max_trajectories_per_scene is not None
                    and traj_count >= self.max_trajectories_per_scene
                ):
                    break

        return episodes

    def __len__(self) -> int:
        return len(self.episode_infos)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for episode_info in self.episode_infos:
            yield self.load_episode(episode_info)

    def load_episode(self, episode_info: Dict[str, str]) -> Dict[str, Any]:
        traj_path = Path(episode_info["path"])
        with open(episode_info["metadata_path"], "r") as f:
            metadata = json.load(f)

        traj_version = "augmented" if self.use_augmented else "non_augmented"
        traj_version_dir = traj_path / traj_version

        # RGB frames are optional: a poses-only dataset has no rendered pixels, so the
        # runner materializes them on the fly from the poses (+ stored rotations).
        rgb_dir = traj_version_dir / "rgb"
        rgb_files = sorted(
            f for f in rgb_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ) if rgb_dir.is_dir() else []
        rgb_images = [Image.open(f).convert("RGB") for f in rgb_files] if rgb_files else None

        poses = []
        rotations = []
        aug_rel_rotations = []
        pose_files = sorted(f for f in (traj_version_dir / "pose").iterdir() if f.suffix == ".json")
        for pose_file in pose_files:
            with open(pose_file, "r") as f:
                pose = json.load(f)
            poses.append(pose["position"])
            rotations.append(habitat_sim.utils.common.quat_from_coeffs(pose["rotation"]))
            aug_rel_rotations.append(pose.get("aug_rel_rotation_quat"))

        query_dir = Path(episode_info["query_dir"])
        with open(query_dir / "pose.json", "r") as f:
            query_pose = json.load(f)

        goal_point_id = None
        if self.tracking_mode == "goal_point":
            with open(query_dir / "query_metadata.json", "r") as f:
                goal_point_id = json.load(f)["goal_point_id"]

        query_rgb = query_dir / "rgb.jpg"

        return {
            "scene_id": episode_info["scene_id"],
            "trajectory_id": episode_info["trajectory_id"],
            "query_id": episode_info["query_id"],
            "metadata": metadata,
            "rgb_images": rgb_images,
            "poses": np.asarray(poses, dtype=np.float32),
            "rotations": rotations,
            "aug_rel_rotations": aug_rel_rotations,
            "traj_version": traj_version,
            "query_pose": query_pose,
            "query_image": Image.open(query_rgb).convert("RGB") if query_rgb.is_file() else None,
            "trajectory_path": traj_path,
            "tracking_mode": self.tracking_mode,
            "goal_point_id": goal_point_id,
        }
