"""Configuration loading for navigation evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class SceneDatasetConfig:
    name: str
    data_path: str
    scene_dataset_config: str
    scenes_directory: str
    scene_pattern: str
    needs_lighting: bool = False


@dataclass
class SimulatorConfig:
    scene_datasets: List[SceneDatasetConfig]
    width: int = 512
    height: int = 512
    camera_fov: List[float] = field(default_factory=lambda: [80.0, 140.0])
    sensor_height: List[float] = field(default_factory=lambda: [0.4, 1.7])
    enable_physics: bool = False
    control_hz: float = 10.0
    max_linear_velocity: float = 0.5
    max_angular_velocity: float = 1.2
    random_seed: int = 42


@dataclass
class EvaluationConfig:
    dataset_path: str
    log_dir: str = "eval_results"
    scenes_to_eval: Optional[List[str]] = None
    max_trajectories_per_scene: Optional[int] = None
    max_episodes: Optional[int] = None
    max_queries_per_trajectory: Optional[int] = 2
    use_augmented: bool = True
    query_type: str = "off_traj"
    tracking_mode: str = "forward"
    max_steps: int = 1000
    success_distance_threshold: float = 0.5
    change_height: bool = True
    change_camera: bool = True
    save_trajectories: bool = True


@dataclass
class MethodConfig:
    name: str = "lotis"
    config_path: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunConfig:
    simulator: SimulatorConfig
    evaluation: EvaluationConfig
    method: MethodConfig


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_config(path: str | Path) -> RunConfig:
    raw = load_yaml(path)

    sim_raw = raw["simulator"]
    scene_datasets = [
        SceneDatasetConfig(**scene_raw)
        for scene_raw in sim_raw.get("scene_datasets", [])
    ]
    if not scene_datasets:
        raise ValueError("Config must define at least one simulator.scene_datasets entry")

    eval_raw = raw["evaluation"]
    method_raw = raw.get("method", {})

    return RunConfig(
        simulator=SimulatorConfig(
            scene_datasets=scene_datasets,
            width=sim_raw.get("width", 512),
            height=sim_raw.get("height", 512),
            camera_fov=sim_raw.get("camera_fov", [80.0, 140.0]),
            sensor_height=sim_raw.get("sensor_height", [0.4, 1.7]),
            enable_physics=sim_raw.get("enable_physics", False),
            control_hz=sim_raw.get("control_hz", 10.0),
            max_linear_velocity=sim_raw.get("max_linear_velocity", 0.5),
            max_angular_velocity=sim_raw.get("max_angular_velocity", 1.2),
            random_seed=sim_raw.get("random_seed", 42),
        ),
        evaluation=EvaluationConfig(**eval_raw),
        method=MethodConfig(
            name=method_raw.get("name", "lotis"),
            config_path=method_raw.get("config_path"),
            params=method_raw.get("params", {}),
        ),
    )


def load_method_config(config: RunConfig) -> Dict[str, Any]:
    params = dict(config.method.params)
    if config.method.config_path is None:
        return params

    method_path = Path(config.method.config_path)
    file_params = load_yaml(method_path)
    file_params.update(params)
    return file_params
