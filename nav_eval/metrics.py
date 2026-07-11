"""Metrics and result serialization for navigation evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class EpisodeResult:
    scene_id: str
    trajectory_id: str
    query_id: str
    success: bool
    num_steps: int
    final_distance_to_goal: float
    path_length: float
    geodesic_distance: float
    spl: float
    collision: bool
    timeout: bool
    time_elapsed: float
    execution_path: List[np.ndarray] = field(default_factory=list)
    velocities: List[tuple] = field(default_factory=list)
    mean_predict_time: float = 0.0
    mean_control_time: float = 0.0
    mean_total_time: float = 0.0
    control_hz: float = 0.0
    overall_hz: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "trajectory_id": self.trajectory_id,
            "query_id": self.query_id,
            "success": bool(self.success),
            "num_steps": int(self.num_steps),
            "final_distance_to_goal": float(self.final_distance_to_goal),
            "path_length": float(self.path_length),
            "geodesic_distance": float(self.geodesic_distance),
            "spl": float(self.spl),
            "collision": bool(self.collision),
            "timeout": bool(self.timeout),
            "time_elapsed": float(self.time_elapsed),
            "execution_path": [p.tolist() if hasattr(p, "tolist") else list(p) for p in self.execution_path],
            "velocities": [[float(v[0]), float(v[1])] for v in self.velocities],
            "mean_predict_time": float(self.mean_predict_time),
            "mean_control_time": float(self.mean_control_time),
            "mean_total_time": float(self.mean_total_time),
            "control_hz": float(self.control_hz),
            "overall_hz": float(self.overall_hz),
        }


class EvaluationMetrics:
    """Tracks episode metrics and writes `results.json`."""

    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[EpisodeResult] = []
        self.current: Optional[Dict[str, Any]] = None

    def start_episode(
        self,
        scene_id: str,
        trajectory_id: str,
        query_id: str,
        goal_position: np.ndarray,
        geodesic_distance: float,
    ) -> None:
        self.current = {
            "scene_id": scene_id,
            "trajectory_id": trajectory_id,
            "query_id": query_id,
            "goal_position": goal_position,
            "geodesic_distance": geodesic_distance,
            "positions": [],
            "velocities": [],
            "num_steps": 0,
            "collision": False,
            "predict_times": [],
            "control_times": [],
            "total_times": [],
        }

    def update_step(
        self,
        position: np.ndarray,
        linear_velocity: float,
        angular_velocity: float,
        collision: bool = False,
        predict_time: float = 0.0,
        control_time: float = 0.0,
        total_time: float = 0.0,
    ) -> None:
        if self.current is None:
            raise RuntimeError("No active episode")

        self.current["positions"].append(position.copy())
        self.current["velocities"].append((linear_velocity, angular_velocity))
        self.current["num_steps"] += 1
        self.current["predict_times"].append(predict_time)
        self.current["control_times"].append(control_time)
        self.current["total_times"].append(total_time)
        if collision:
            self.current["collision"] = True

    def end_episode(
        self,
        final_position: np.ndarray,
        time_elapsed: float,
        success: bool,
        timeout: bool = False,
    ) -> EpisodeResult:
        if self.current is None:
            raise RuntimeError("No active episode")

        positions = np.asarray(self.current["positions"])
        path_length = float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))) if len(positions) > 1 else 0.0
        goal_position = self.current["goal_position"]
        final_distance = float(np.linalg.norm(final_position - goal_position))
        geodesic = float(self.current["geodesic_distance"])
        spl = geodesic / max(path_length, geodesic) if success and path_length > 0 else 0.0

        predict_times = self.current["predict_times"]
        control_times = self.current["control_times"]
        total_times = self.current["total_times"]
        mean_predict = float(np.mean(predict_times)) if predict_times else 0.0
        mean_control = float(np.mean(control_times)) if control_times else 0.0
        mean_total = float(np.mean(total_times)) if total_times else 0.0
        control_hz = 1.0 / mean_total if mean_total > 0 else 0.0
        overall_hz = self.current["num_steps"] / time_elapsed if time_elapsed > 0 else 0.0

        result = EpisodeResult(
            scene_id=self.current["scene_id"],
            trajectory_id=self.current["trajectory_id"],
            query_id=self.current["query_id"],
            success=success,
            num_steps=self.current["num_steps"],
            final_distance_to_goal=final_distance,
            path_length=path_length,
            geodesic_distance=geodesic,
            spl=spl,
            collision=self.current["collision"],
            timeout=timeout,
            time_elapsed=time_elapsed,
            execution_path=self.current["positions"],
            velocities=self.current["velocities"],
            mean_predict_time=mean_predict,
            mean_control_time=mean_control,
            mean_total_time=mean_total,
            control_hz=control_hz,
            overall_hz=overall_hz,
        )
        self.results.append(result)
        self.current = None
        return result

    def get_summary(self) -> Dict[str, float]:
        if not self.results:
            return {}

        successful = [r for r in self.results if r.success]
        return {
            "num_trajectories": len(self.results),
            "success_rate": float(np.mean([r.success for r in self.results])),
            "mean_spl": float(np.mean([r.spl for r in self.results])),
            "success_spl": float(np.mean([r.spl for r in successful])) if successful else 0.0,
            "collision_rate": float(np.mean([r.collision for r in self.results])),
            "timeout_rate": float(np.mean([r.timeout for r in self.results])),
            "mean_path_length": float(np.mean([r.path_length for r in self.results])),
            "mean_steps": float(np.mean([r.num_steps for r in self.results])),
            "mean_predict_time_ms": float(np.mean([r.mean_predict_time for r in self.results]) * 1000),
            "mean_control_time_ms": float(np.mean([r.mean_control_time for r in self.results]) * 1000),
            "mean_total_time_ms": float(np.mean([r.mean_total_time for r in self.results]) * 1000),
            "mean_control_hz": float(np.mean([r.control_hz for r in self.results])),
            "mean_overall_hz": float(np.mean([r.overall_hz for r in self.results])),
        }

    def print_progress(self) -> None:
        summary = self.get_summary()
        if not summary:
            return
        print("\n" + "-" * 60)
        print(f"PROGRESS UPDATE: {summary['num_trajectories']} episodes completed")
        print("-" * 60)
        print(f"Success rate:  {summary['success_rate']:.2%}  |  SPL: {summary['mean_spl']:.3f}")
        print(f"Control Hz: {summary['mean_control_hz']:.1f} (method only)  |  Overall Hz: {summary['mean_overall_hz']:.1f} (full iteration)")
        print("-" * 60 + "\n")

    def print_summary(self) -> None:
        summary = self.get_summary()
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Trajectories evaluated: {summary['num_trajectories']}")
        print(f"Success rate:          {summary['success_rate']:.2%}")
        print(f"Mean SPL:              {summary['mean_spl']:.3f}")
        print(f"Success SPL:           {summary['success_spl']:.3f}")
        print(f"Collision rate:        {summary['collision_rate']:.2%}")
        print(f"Timeout rate:          {summary['timeout_rate']:.2%}")
        print(f"Mean path length:      {summary['mean_path_length']:.2f} m")
        print(f"Mean steps:            {summary['mean_steps']:.1f}")
        print()
        print("TIMING STATISTICS")
        print("-" * 60)
        print(f"Mean predict time:     {summary['mean_predict_time_ms']:.2f} ms")
        print(f"Mean control time:     {summary['mean_control_time_ms']:.2f} ms")
        print(f"Mean total time:       {summary['mean_total_time_ms']:.2f} ms")
        print(f"Mean control freq:     {summary['mean_control_hz']:.1f} Hz (method only)")
        print(f"Mean overall freq:     {summary['mean_overall_hz']:.1f} Hz (full iteration)")
        print("=" * 60)

    def save_results(self, filename: str = "results.json") -> None:
        output_path = self.log_dir / filename
        with open(output_path, "w") as f:
            json.dump(
                {
                    "summary": self.get_summary(),
                    "trajectories": [result.to_dict() for result in self.results],
                },
                f,
                indent=2,
            )
        print(f"Results saved to {output_path}")
