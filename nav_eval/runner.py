"""Single navigation evaluation runner."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

import habitat_sim
import numpy as np
from habitat_sim.sensor import SensorType
from PIL import Image

from .config import RunConfig
from .dataset import NavigationDataset
from .habitat import HabitatRuntime, discover_scenes
from .interfaces import NavigationMethod
from .metrics import EvaluationMetrics


class EvaluationRunner:
    """Runs one registered method over an existing navigation evaluation dataset."""

    def __init__(self, config: RunConfig, method: NavigationMethod):
        self.config = config
        self.method = method
        self.simulator = HabitatRuntime(config.simulator)
        self.scene_infos = discover_scenes(config.simulator.scene_datasets)
        print(f"Available scenes: {len(self.scene_infos)}")

        eval_cfg = config.evaluation
        self.dataset = NavigationDataset(
            dataset_path=eval_cfg.dataset_path,
            scenes_to_eval=eval_cfg.scenes_to_eval,
            max_trajectories_per_scene=eval_cfg.max_trajectories_per_scene,
            max_episodes=eval_cfg.max_episodes,
            max_queries_per_trajectory=eval_cfg.max_queries_per_trajectory,
            use_augmented=eval_cfg.use_augmented,
            query_type=eval_cfg.query_type,
            tracking_mode=eval_cfg.tracking_mode,
        )
        self.metrics = EvaluationMetrics(log_dir=eval_cfg.log_dir)
        self.dt = 1.0 / config.simulator.control_hz
        self.current_fov = (config.simulator.camera_fov[0] + config.simulator.camera_fov[1]) / 2.0
        self.camera_width = config.simulator.width
        self.camera_height = config.simulator.height
        self.current_sensor_height = (config.simulator.sensor_height[0] + config.simulator.sensor_height[1]) / 2.0
        self.visualize = False
        self._viz_step = 0
        self.materialize = False

    def run(self, start_idx: int = 0, on_episode=None, visualize: bool = False, materialize: bool = False) -> None:
        self.visualize = visualize
        self.materialize = materialize
        eval_cfg = self.config.evaluation
        print("\n" + "=" * 60)
        print("STARTING EVALUATION")
        print("=" * 60)
        print(f"Method: {self.config.method.name}")
        print(f"Dataset: {eval_cfg.dataset_path}")
        print(f"Query type: {eval_cfg.query_type}")
        print(f"Tracking mode: {self._tracking_mode_label(eval_cfg.tracking_mode)}")
        print(f"Trajectories: {len(self.dataset)}")
        print(f"Max steps per episode: {eval_cfg.max_steps}")
        if start_idx > 0:
            print(f"Starting from index: {start_idx}")
        print("=" * 60 + "\n")

        print("Setting up navigation method...")
        self.method.setup()
        self.method.set_simulator(self.simulator)
        print("Method setup complete.\n")

        try:
            for idx, episode in enumerate(self.dataset):
                if idx < start_idx:
                    continue
                print(f"\n[{idx + 1}/{len(self.dataset)}] Evaluating trajectory: {episode['scene_id']}/{episode['trajectory_id']}")
                result = self.evaluate_episode(episode)
                status = "SUCCESS" if result.success else "FAILED"
                print(f"  Result: {status} | Steps: {result.num_steps} | Path length: {result.path_length:.2f}m | SPL: {result.spl:.3f}")
                if result.timeout:
                    print("  Warning: Timeout")
                if (idx + 1) % 10 == 0:
                    self.metrics.print_progress()
                if on_episode is not None:
                    on_episode(idx + 1, len(self.dataset), self.metrics.get_summary(), result)

            print("\n" + "=" * 60)
            print("EVALUATION COMPLETE")
            print("=" * 60)
            self.metrics.print_summary()
            self.metrics.save_results()
        finally:
            self.method.cleanup()
            self.simulator.close()

    def evaluate_episode(self, episode: Dict[str, Any]):
        eval_start = time.time()
        scene_id = episode["metadata"]["scene_id"]
        scene_info = next((s for s in self.scene_infos if s["scene_id"] == scene_id), None)
        if scene_info is None:
            raise ValueError(f"Scene '{scene_id}' not found in configured scene datasets")

        scene_load_start = time.time()
        if (
            self.simulator.sim is None
            or self.simulator.sim.config.sim_cfg.scene_id != scene_info["scene_path"]
        ):
            print(f"  Loading scene: {scene_id}")
            self.simulator.initialize(
                scene_info["scene_path"],
                scene_info["scene_dataset_config"],
                needs_lighting=scene_info["needs_lighting"],
            )
            scene_load_time = time.time() - scene_load_start
            print(f"  Scene load time: {scene_load_time:.2f}s")
        else:
            scene_load_time = 0.0

        # Materialize the reference trajectory frames from poses when the pixels are
        # absent (poses-only dataset) or when forced. With visualization on, stream
        # both the on-disk and freshly-rendered frames so they can be compared.
        disk_rgb = episode.get("rgb_images")
        if disk_rgb is None or self.materialize:
            rendered = self._materialize_trajectory(episode)
            if self.visualize and disk_rgb is not None:
                self._log_trajectory_comparison(disk_rgb, rendered)
            episode["rgb_images"] = rendered

        # Hand the method only its legitimate inputs: the reference frames and the
        # task spec. The ground-truth trajectory geometry stays with the runner.
        self.method.reset_episode({
            "scene_id": episode["scene_id"],
            "trajectory_id": episode["trajectory_id"],
            "query_id": episode["query_id"],
            "rgb_images": episode["rgb_images"],
            "tracking_mode": episode["tracking_mode"],
            "goal_point_id": episode["goal_point_id"],
        })

        tracking_mode = episode["tracking_mode"]
        if tracking_mode == "backward":
            goal_position = episode["poses"][0]
        elif tracking_mode == "goal_point":
            goal_position = episode["poses"][episode["goal_point_id"]]
        else:
            goal_position = episode["poses"][-1]

        if self.config.evaluation.change_camera:
            camera_params = episode["query_pose"]["camera_params"]
        else:
            camera_params = episode["metadata"]["camera_params"]

        if self.config.evaluation.change_height:
            sensor_height = episode["query_pose"]["camera_params"]["height"]
        else:
            sensor_height = episode["metadata"]["camera_params"]["height"]
        print(f"  Using sensor height: {sensor_height:.2f}m")

        self.current_sensor_height = sensor_height
        self.current_fov = camera_params["fov"]
        self.current_aspect_ratio = camera_params["aspect_ratio"]
        self.simulator.update_camera_params(
            agent_id=0,
            fov=camera_params["fov"],
            height=sensor_height,
            aspect_ratio=camera_params["aspect_ratio"],
        )
        self._update_camera_resolution()

        start_position = np.asarray(episode["query_pose"]["position"], dtype=np.float32)
        start_rotation = habitat_sim.utils.common.quat_from_coeffs(episode["query_pose"]["rotation"])
        agent_state = habitat_sim.AgentState()
        agent_state.position = start_position
        agent_state.rotation = start_rotation
        self.simulator.sim.agents[0].set_state(agent_state)

        path_points, geodesic_distance = self.simulator.compute_path(start_position, goal_position)
        if path_points is None:
            print("Warning: no navigable path found from start to goal; using Euclidean distance")
            geodesic_distance = float(np.linalg.norm(goal_position - start_position))

        self.metrics.start_episode(
            scene_id=episode["scene_id"],
            trajectory_id=episode["trajectory_id"],
            query_id=episode["query_id"],
            goal_position=goal_position,
            geodesic_distance=geodesic_distance,
        )

        if self.visualize:
            import rerun as rr

            rgb_images = episode["rgb_images"]
            if tracking_mode == "backward":
                goal_idx = 0
            elif tracking_mode == "goal_point":
                goal_idx = episode["goal_point_id"]
            else:
                goal_idx = len(rgb_images) - 1
            rr.set_time("step", sequence=self._viz_step)
            rr.log("goal", rr.Image(np.asarray(rgb_images[goal_idx])).compress(jpeg_quality=90))

        start_time = time.time()
        success = False
        timeout = False

        for step in range(self.config.evaluation.max_steps):
            obs = self._get_observation(step)

            predict_start = time.time()
            action = self.method.act(obs)
            predict_time = time.time() - predict_start
            control_time = 0.0
            total_method_time = predict_time + control_time

            if step % 50 == 0:
                hz = 1.0 / total_method_time if total_method_time > 0 else 0.0
                print(f"  Step {step}: predict={predict_time*1000:.1f}ms, control={control_time*1000:.1f}ms, total={total_method_time*1000:.1f}ms ({hz:.1f}Hz)")

            if self.visualize:
                import rerun as rr

                rr.set_time("step", sequence=self._viz_step)
                rr.log("camera", rr.Image(obs["rgb"]).compress(jpeg_quality=90))
                self.method.visualize(obs)
                self._viz_step += 1

            linear_velocity = np.clip(
                action.linear_velocity,
                -self.config.simulator.max_linear_velocity,
                self.config.simulator.max_linear_velocity,
            )
            angular_velocity = np.clip(
                action.angular_velocity,
                -self.config.simulator.max_angular_velocity,
                self.config.simulator.max_angular_velocity,
            )
            self.simulator.step_velocity(linear_velocity, angular_velocity, self.dt)

            current_position = self.simulator.sim.agents[0].state.position
            self.metrics.update_step(
                position=current_position,
                linear_velocity=linear_velocity,
                angular_velocity=angular_velocity,
                collision=False,
                predict_time=predict_time,
                control_time=control_time,
                total_time=total_method_time,
            )

            if np.linalg.norm(current_position - goal_position) < self.config.evaluation.success_distance_threshold:
                success = True
                break

        if not success and step >= self.config.evaluation.max_steps - 1:
            timeout = True

        time_elapsed = time.time() - start_time
        final_position = self.simulator.sim.agents[0].state.position
        metrics_start = time.time()
        result = self.metrics.end_episode(
            final_position=final_position,
            time_elapsed=time_elapsed,
            success=success,
            timeout=timeout,
        )
        metrics_time = time.time() - metrics_start

        total_eval_time = time.time() - eval_start
        setup_time = total_eval_time - scene_load_time - time_elapsed - metrics_time
        print(f"\n  TIMING BREAKDOWN (Total: {total_eval_time:.2f}s):")
        print(f"    Scene loading:    {scene_load_time:6.2f}s ({scene_load_time/total_eval_time*100:5.1f}%)")
        print(f"    Setup/init:       {setup_time:6.2f}s ({setup_time/total_eval_time*100:5.1f}%)")
        print(f"    Navigation loop:  {time_elapsed:6.2f}s ({time_elapsed/total_eval_time*100:5.1f}%)")
        print(f"      - Avg predict:  {result.mean_predict_time*1000:6.1f}ms/step")
        print(f"      - Avg control:  {result.mean_control_time*1000:6.1f}ms/step")
        print(f"      - Control freq: {result.control_hz:6.1f}Hz (method only)")
        print(f"      - Overall freq: {result.overall_hz:6.1f}Hz (full iteration)")
        print(f"    Metrics/cleanup:  {metrics_time:6.2f}s ({metrics_time/total_eval_time*100:5.1f}%)")
        return result

    def _materialize_trajectory(self, episode: Dict[str, Any]):
        """Render the reference trajectory frames from poses (+ stored augmentation rotations)."""
        cp = episode["metadata"]["camera_params"]
        self.simulator.update_camera_params(
            agent_id=0, fov=cp["fov"], height=cp["height"], aspect_ratio=cp["aspect_ratio"]
        )
        rels = episode.get("aug_rel_rotations") or [None] * len(episode["rotations"])
        images = []
        for position, rotation, rel in zip(episode["poses"], episode["rotations"], rels):
            state = habitat_sim.AgentState()
            state.position = np.asarray(position, dtype=np.float32)
            state.rotation = rotation
            self.simulator.sim.agents[0].set_state(state)
            self.simulator.reset_camera_rotation(0)
            if rel is not None:
                self.simulator.update_rel_camera_rotation(
                    0, habitat_sim.utils.common.quat_from_coeffs(np.asarray(rel, dtype=np.float32))
                )
            obs = self.simulator.get_observations([0], [SensorType.COLOR])[0]
            images.append(Image.fromarray(obs["color_sensor"][:, :, :3]))
        return images

    def _log_trajectory_comparison(self, disk_images, rendered_images) -> None:
        import rerun as rr

        for i, (disk_img, rendered_img) in enumerate(zip(disk_images, rendered_images)):
            rr.set_time("traj_frame", sequence=i)
            rr.log("trajectory/disk", rr.Image(np.asarray(disk_img)).compress(jpeg_quality=90))
            rr.log("trajectory/rendered", rr.Image(np.asarray(rendered_img)).compress(jpeg_quality=90))

    def _get_observation(self, step: int) -> Dict[str, Any]:
        obs = self.simulator.get_observations([0], [SensorType.COLOR])[0]
        rgb = obs["color_sensor"][:, :, :3]
        agent_state = self.simulator.sim.agents[0].state

        fx = (self.camera_width / 2.0) / np.tan(np.deg2rad(self.current_fov) / 2.0)
        fy = fx
        cx = self.camera_width / 2.0
        cy = self.camera_height / 2.0
        intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

        return {
            "rgb": rgb,
            "rgb_pil": Image.fromarray(rgb),
            "intrinsics": intrinsics,
            "position": agent_state.position.copy(),
            "rotation": agent_state.rotation,
            "step": step,
        }

    def _update_camera_resolution(self) -> None:
        for sensor in self.simulator.sim.get_agent(0).agent_config.sensor_specifications:
            if isinstance(sensor, habitat_sim.sensor.CameraSensorSpec) and sensor.uuid == "color_sensor":
                self.camera_width = sensor.resolution[1]
                self.camera_height = sensor.resolution[0]
                break

    @staticmethod
    def _tracking_mode_label(mode: str) -> str:
        if mode == "forward":
            return "Forward (goal at END)"
        if mode == "backward":
            return "Backward (goal at START)"
        if mode == "goal_point":
            return "Goal Point (goal at specific trajectory point)"
        return mode
