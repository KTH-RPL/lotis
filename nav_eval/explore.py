"""Explore an evaluation episode: render its reference trajectory and query view.

Because the Habitat scene RGB cannot be redistributed, the evaluation dataset ships
poses only and the frames are generated on the fly from your local scenes. This tool
renders one episode (the reference trajectory frames + the query view + the goal frame)
and streams them to a Rerun viewer so you can inspect the tasks.

    uv run python -m nav_eval.explore --dataset gibson --start on --direction forward \
        --camera cross --index 0
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import habitat_sim
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from habitat_sim.sensor import SensorType
from habitat_sim.utils.common import quat_from_coeffs

from .config import load_config
from .dataset import NavigationDataset
from .habitat import HabitatRuntime, discover_scenes
from .run import CAMERA_TO_AUGMENTED, DATASET_CONFIGS, DIRECTION_TO_TRACKING_MODE, START_TO_QUERY_TYPE


def _render_color(sim: HabitatRuntime) -> np.ndarray:
    return sim.get_observations([0], [SensorType.COLOR])[0]["color_sensor"][:, :, :3]


def _render_trajectory(sim: HabitatRuntime, episode) -> list:
    cp = episode["metadata"]["camera_params"]
    sim.update_camera_params(0, fov=cp["fov"], height=cp["height"], aspect_ratio=cp["aspect_ratio"])
    rels = episode.get("aug_rel_rotations") or [None] * len(episode["rotations"])
    frames = []
    for position, rotation, rel in zip(episode["poses"], episode["rotations"], rels):
        state = habitat_sim.AgentState()
        state.position = np.asarray(position, dtype=np.float32)
        state.rotation = rotation
        sim.sim.agents[0].set_state(state)
        sim.reset_camera_rotation(0)
        if rel is not None:
            sim.update_rel_camera_rotation(0, quat_from_coeffs(np.asarray(rel, dtype=np.float32)))
        frames.append(_render_color(sim).copy())
    return frames


def _render_query(sim: HabitatRuntime, episode) -> np.ndarray:
    cp = episode["query_pose"]["camera_params"]
    sim.update_camera_params(0, fov=cp["fov"], height=cp["height"], aspect_ratio=cp["aspect_ratio"])
    state = habitat_sim.AgentState()
    state.position = np.asarray(episode["query_pose"]["position"], dtype=np.float32)
    state.rotation = quat_from_coeffs(episode["query_pose"]["rotation"])
    sim.sim.agents[0].set_state(state)
    return _render_color(sim).copy()


def _blueprint() -> rrb.Blueprint:
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial2DView(name="Query", origin="query"),
            rrb.Spatial2DView(name="Reference trajectory", origin="reference_trajectory"),
            rrb.Spatial2DView(name="Goal", origin="goal"),
        ),
        collapse_panels=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=sorted(DATASET_CONFIGS), default=None, help="Built-in dataset config")
    parser.add_argument("--config", default=None, help="Path to a dataset/evaluation YAML config")
    parser.add_argument("--scene-data-dir", default=None, help="Root containing scene_datasets/ (env: HABITAT_DATA_DIR)")
    parser.add_argument("--start", choices=["on", "off"], default="on")
    parser.add_argument("--direction", choices=["forward", "backward", "any"], default="forward")
    parser.add_argument("--camera", choices=["matched", "cross"], default="cross")
    parser.add_argument("--index", type=int, default=0, help="Episode index within the selected case")
    parser.add_argument("--rerun-connect", default=None, metavar="URL", help="Connect to a running Rerun viewer")
    args = parser.parse_args()

    if (args.config is None) == (args.dataset is None):
        parser.error("Provide exactly one of --dataset or --config")

    config = load_config(Path(args.config) if args.config else DATASET_CONFIGS[args.dataset])
    scene_data_dir = args.scene_data_dir or os.environ.get("HABITAT_DATA_DIR")
    if scene_data_dir:
        for scene_dataset in config.simulator.scene_datasets:
            scene_dataset.data_path = scene_data_dir

    dataset = NavigationDataset(
        dataset_path=config.evaluation.dataset_path,
        max_queries_per_trajectory=config.evaluation.max_queries_per_trajectory,
        use_augmented=CAMERA_TO_AUGMENTED[args.camera],
        query_type=START_TO_QUERY_TYPE[args.start],
        tracking_mode=DIRECTION_TO_TRACKING_MODE[args.direction],
    )
    if not 0 <= args.index < len(dataset.episode_infos):
        raise SystemExit(f"--index {args.index} out of range (0..{len(dataset.episode_infos) - 1})")
    episode = dataset.load_episode(dataset.episode_infos[args.index])

    scenes = discover_scenes(config.simulator.scene_datasets)
    scene = next((s for s in scenes if s["scene_id"] == episode["metadata"]["scene_id"]), None)
    if scene is None:
        raise SystemExit(
            f"Scene '{episode['metadata']['scene_id']}' not found. "
            f"Point --scene-data-dir (or HABITAT_DATA_DIR) at your Habitat scene_datasets root."
        )

    sim = HabitatRuntime(config.simulator)
    sim.initialize(scene["scene_path"], scene["scene_dataset_config"], scene["needs_lighting"])
    trajectory = _render_trajectory(sim, episode)
    query = _render_query(sim, episode)

    tracking_mode = episode["tracking_mode"]
    if tracking_mode == "backward":
        goal_idx = 0
    elif tracking_mode == "goal_point":
        goal_idx = episode["goal_point_id"]
    else:
        goal_idx = len(trajectory) - 1

    rr.init("nav_eval_explore")
    if args.rerun_connect:
        rr.connect_grpc(args.rerun_connect)
    else:
        rr.spawn()
    rr.send_blueprint(_blueprint())
    rr.log("query", rr.Image(query).compress(jpeg_quality=90), static=True)
    rr.log("goal", rr.Image(np.asarray(trajectory[goal_idx])).compress(jpeg_quality=90), static=True)
    for i, frame in enumerate(trajectory):
        rr.set_time("frame", sequence=i)
        rr.log("reference_trajectory", rr.Image(frame).compress(jpeg_quality=90))

    print(f"Episode {args.index}: {episode['scene_id']}/{episode['trajectory_id']}/{episode['query_id']}  "
          f"| tracking={tracking_mode} | {len(trajectory)} reference frames | goal frame={goal_idx}")
    sim.close()


if __name__ == "__main__":
    main()
