"""Small Habitat-Sim wrapper for evaluation only."""

from __future__ import annotations

import glob
import os
import random
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import habitat_sim
import numpy as np
from habitat_sim.sensor import SensorType

from ..config import SceneDatasetConfig, SimulatorConfig


def discover_scenes(scene_datasets: List[SceneDatasetConfig]) -> List[Dict[str, Any]]:
    """Discover Habitat scene files from configured scene datasets."""
    scenes = []
    for dataset in scene_datasets:
        scenes_dir = os.path.join(dataset.data_path, dataset.scenes_directory)
        scene_dataset_config = os.path.join(dataset.data_path, dataset.scene_dataset_config)
        if not os.path.isdir(scenes_dir):
            print(f"Warning: scenes directory not found: {scenes_dir}")
            continue

        pattern = os.path.join(scenes_dir, dataset.scene_pattern)
        for scene_file in sorted(glob.glob(pattern, recursive=True)):
            rel_path = os.path.relpath(scene_file, scenes_dir)
            scene_id = os.path.splitext(rel_path)[0]
            scenes.append(
                {
                    "dataset_name": dataset.name,
                    "scene_path": scene_file,
                    "scene_dataset_config": scene_dataset_config,
                    "scene_id": scene_id,
                    "needs_lighting": dataset.needs_lighting,
                }
            )
    return scenes


class HabitatRuntime:
    """Runtime wrapper for loading scenes, reading sensors, and stepping an agent."""

    def __init__(self, config: SimulatorConfig):
        self.config = config
        self.sim: Optional[habitat_sim.Simulator] = None
        self.scene_settings: Dict[str, Any] = {}

    def initialize(self, scene_path: str, scene_dataset_config: str, needs_lighting: bool) -> bool:
        if self.sim is not None:
            self.close()

        self.scene_settings = {
            "scene_path": scene_path,
            "scene_dataset_config": scene_dataset_config,
            "needs_lighting": needs_lighting,
        }
        try:
            self.sim = habitat_sim.Simulator(self._make_cfg())
            random.seed(self.config.random_seed)
            self.sim.seed(self.config.random_seed)
            return self.load_navmesh()
        except Exception as exc:
            print(f"Failed to initialize simulator: {exc}")
            self.sim = None
            return False

    def close(self) -> None:
        if self.sim is not None:
            self.sim.close()
            self.sim = None

    def load_navmesh(self) -> bool:
        if self.sim is None:
            raise RuntimeError("Simulator is not initialized")
        self.sim.navmesh_visualization = False
        navmesh_settings = habitat_sim.NavMeshSettings()
        navmesh_settings.set_defaults()
        navmesh_settings.include_static_objects = True
        ok = self.sim.recompute_navmesh(self.sim.pathfinder, navmesh_settings)
        if not ok:
            print("Failed to load navmesh.")
        return ok

    def compute_path(self, start_point: np.ndarray, end_point: np.ndarray):
        if self.sim is None:
            raise RuntimeError("Simulator is not initialized")
        path = habitat_sim.ShortestPath()
        path.requested_start = start_point
        path.requested_end = end_point
        if not self.sim.pathfinder.find_path(path):
            return None, 0.0
        return path.points, path.geodesic_distance

    def update_camera_params(self, agent_id: int, fov: float, height: float, aspect_ratio: float) -> None:
        if self.sim is None:
            raise RuntimeError("Simulator is not initialized")

        new_width = self.config.width
        new_height = self.config.height
        if aspect_ratio < 1.0:
            new_height = int(new_width / aspect_ratio)
        else:
            new_width = int(new_height * aspect_ratio)

        agent_config = self.sim.get_agent(agent_id).agent_config

        for _, sensor in self.sim._Simulator__sensors[agent_id].items():
            habitat_sim.SensorFactory.delete_sensor(sensor._sensor_object)

        for sensor in agent_config.sensor_specifications:
            if isinstance(sensor, habitat_sim.sensor.CameraSensorSpec):
                sensor.hfov = fov
                sensor.position[1] = height
                sensor.resolution = [new_height, new_width]

        self.sim.get_agent(agent_id).reconfigure(agent_config, reconfigure_sensors=True)

        for sensor in agent_config.sensor_specifications:
            if isinstance(sensor, habitat_sim.sensor.CameraSensorSpec):
                self.sim._update_simulator_sensors(sensor.uuid, agent_id)

    def reset_camera_rotation(self, agent_id: int) -> None:
        if self.sim is None:
            raise RuntimeError("Simulator is not initialized")
        agent_state = self.sim.get_agent(agent_id).state
        self.sim.agents[agent_id].set_state(agent_state, infer_sensor_states=True)

    def update_rel_camera_rotation(self, agent_id: int, rotation) -> None:
        """Rotate the camera sensor relative to the agent's current orientation."""
        if self.sim is None:
            raise RuntimeError("Simulator is not initialized")
        agent_state = self.sim.get_agent(agent_id).state
        abs_rotation = agent_state.rotation * rotation
        self.sim.agents[agent_id].set_state(agent_state, infer_sensor_states=True)
        for sensor_state in agent_state.sensor_states.values():
            if hasattr(sensor_state, "rotation"):
                sensor_state.rotation = abs_rotation
        self.sim.agents[agent_id].set_state(agent_state, infer_sensor_states=False)

    def get_observations(self, agent_ids: List[int], sensor_types: Optional[List[SensorType]] = None):
        if self.sim is None:
            raise RuntimeError("Simulator is not initialized")

        observations: Dict[int, Dict[str, Any]] = OrderedDict()
        for agent_id in agent_ids:
            for _, sensor in self.sim._Simulator__sensors[agent_id].items():
                if sensor_types is None or sensor._spec.sensor_type in sensor_types:
                    sensor.draw_observation()

        for agent_id in agent_ids:
            agent_observations = {}
            for sensor_uuid, sensor in self.sim._Simulator__sensors[agent_id].items():
                if sensor_types is None or sensor._spec.sensor_type in sensor_types:
                    agent_observations[sensor_uuid] = sensor.get_observation()
            observations[agent_id] = agent_observations
        return observations

    def step_velocity(self, linear_velocity: float, angular_velocity: float, dt: float) -> None:
        if self.sim is None:
            raise RuntimeError("Simulator is not initialized")
        if linear_velocity == 0 and angular_velocity == 0:
            return

        agent_state = self.sim.agents[0].state
        vel_control = habitat_sim.physics.VelocityControl()
        vel_control.controlling_lin_vel = True
        vel_control.lin_vel_is_local = True
        vel_control.controlling_ang_vel = True
        vel_control.ang_vel_is_local = True
        vel_control.linear_velocity = np.array([0, 0, -linear_velocity])
        vel_control.angular_velocity = np.array([0, angular_velocity, 0])

        previous = habitat_sim.RigidState(
            habitat_sim.utils.common.quat_to_magnum(agent_state.rotation),
            agent_state.position,
        )
        target = vel_control.integrate_transform(dt, previous)
        end_pos = self.sim.step_filter(previous.translation, target.translation)

        agent_state.position = end_pos
        agent_state.rotation = habitat_sim.utils.common.quat_from_magnum(target.rotation)
        self.sim.agents[0].set_state(agent_state, reset_sensors=False)

    def _make_cfg(self) -> habitat_sim.Configuration:
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.gpu_device_id = 0
        sim_cfg.scene_id = self.scene_settings["scene_path"]
        sim_cfg.scene_dataset_config_file = self.scene_settings["scene_dataset_config"]
        sim_cfg.enable_physics = self.config.enable_physics
        sim_cfg.allow_sliding = True
        sim_cfg.load_semantic_mesh = False
        sim_cfg.use_semantic_textures = False
        if self.scene_settings["needs_lighting"]:
            sim_cfg.override_scene_light_defaults = True
            sim_cfg.scene_light_setup = habitat_sim.gfx.DEFAULT_LIGHTING_KEY

        camera_fov = (self.config.camera_fov[0] + self.config.camera_fov[1]) / 2.0
        sensor_height = (self.config.sensor_height[0] + self.config.sensor_height[1]) / 2.0

        sensor_specs = []
        for uuid, sensor_type in (
            ("color_sensor", habitat_sim.SensorType.COLOR),
            ("depth_sensor", habitat_sim.SensorType.DEPTH),
        ):
            sensor = habitat_sim.CameraSensorSpec()
            sensor.uuid = uuid
            sensor.sensor_type = sensor_type
            sensor.resolution = [self.config.height, self.config.width]
            sensor.position = [0.0, sensor_height, 0.0]
            sensor.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
            sensor.hfov = camera_fov
            sensor_specs.append(sensor)

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = sensor_specs
        agent_cfg.action_space = {
            "move_forward": habitat_sim.agent.ActionSpec(
                "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
            ),
            "turn_left": habitat_sim.agent.ActionSpec(
                "turn_left", habitat_sim.agent.ActuationSpec(amount=30.0)
            ),
            "turn_right": habitat_sim.agent.ActionSpec(
                "turn_right", habitat_sim.agent.ActuationSpec(amount=30.0)
            ),
        }
        return habitat_sim.Configuration(sim_cfg, [agent_cfg])
