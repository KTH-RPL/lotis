"""Command-line entry point for navigation evaluation."""

from __future__ import annotations

import argparse
import contextlib
import copy
import importlib
import os
import shutil
import sys
import traceback
from pathlib import Path

import yaml
from rich import box
from rich.console import Console
from rich.live import Live
from rich.table import Table

from .config import load_config, load_method_config
from .registry import get_method, registered_methods

# Import built-in methods so they register themselves.
from . import methods as _methods  # noqa: F401


DATASET_CONFIGS = {
    "gibson": Path(__file__).resolve().parent / "configs" / "datasets" / "gibson.yaml",
    "hm3d": Path(__file__).resolve().parent / "configs" / "datasets" / "hm3d.yaml",
}


START_TO_QUERY_TYPE = {
    "on": "on_traj",
    "off": "off_traj",
}


DIRECTION_TO_TRACKING_MODE = {
    "forward": "forward",
    "backward": "backward",
    "any": "goal_point",
}


CAMERA_TO_AUGMENTED = {
    "matched": False,
    "cross": True,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run navigation evaluation")
    parser.add_argument("--config", default=None, help="Path to dataset/evaluation YAML config")
    parser.add_argument("--dataset", choices=sorted(DATASET_CONFIGS), default=None, help="Built-in dataset config")
    parser.add_argument("--method", default=None, help="Registered method name; overrides config")
    parser.add_argument("--method-module", default=None, help="Import this module before looking up the method")
    parser.add_argument("--log-dir", default=None, help="Override output directory")
    parser.add_argument("--dataset-path", default=None, help="Override evaluation dataset path")
    parser.add_argument("--scene-data-dir", default=None, help="Root dir containing scene_datasets/; overrides data_path for all scene datasets (env: HABITAT_DATA_DIR)")
    parser.add_argument("--start", choices=["on", "off"], action="append", help="Start type filter. Can be passed multiple times.")
    parser.add_argument("--direction", choices=["forward", "backward", "any"], action="append", help="Direction filter. Can be passed multiple times.")
    parser.add_argument("--camera", choices=["matched", "cross"], action="append", help="Camera/reference condition filter. Can be passed multiple times.")
    parser.add_argument("--max-episodes", type=int, default=None, help="Run only the first N episodes")
    parser.add_argument("--start-idx", type=int, default=0, help="Start from this episode index")
    parser.add_argument("--visualize", action="store_true", help="Stream live RGB, the goal image, and method visualizations to a Rerun viewer")
    parser.add_argument("--rerun-connect", default=None, metavar="URL", help="Connect to an already-running Rerun viewer instead of spawning one (e.g. rerun+http://127.0.0.1:9876/proxy)")
    parser.add_argument("--materialize", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--list-methods", action="store_true", help="List registered methods and exit")
    parser.add_argument("--list-cases", action="store_true", help="List selected evaluation cases and exit")
    args = parser.parse_args()

    if args.method_module is not None:
        importlib.import_module(args.method_module)

    if args.list_methods:
        for name in sorted(registered_methods()):
            print(name)
        return

    if args.config is not None and args.dataset is not None:
        parser.error("Use only one of --config or --dataset")
    if args.config is None and args.dataset is None:
        parser.error("--dataset or --config is required unless --list-methods is used")

    config_path = Path(args.config) if args.config is not None else DATASET_CONFIGS[args.dataset]
    base_config = load_config(config_path)

    scene_data_dir = args.scene_data_dir or os.environ.get("HABITAT_DATA_DIR")
    if scene_data_dir:
        for scene_dataset in base_config.simulator.scene_datasets:
            scene_dataset.data_path = scene_data_dir
    cases = expand_cases(
        starts=args.start,
        directions=args.direction,
        cameras=args.camera,
    )
    if args.list_cases:
        if not cases:
            print("No valid evaluation cases selected")
            return
        for case in cases:
            print(case_name(case))
        return
    if not cases:
        raise ValueError("No valid evaluation cases selected")

    base_log_dir = Path(args.log_dir or base_config.evaluation.log_dir)

    if args.visualize:
        import rerun as rr

        rr.init("nav_eval")
        if args.rerun_connect:
            rr.connect_grpc(args.rerun_connect)
        else:
            rr.spawn()

    # Bind the dashboard to the real terminal *before* any stdout redirection so
    # it keeps rendering there while per-case logs are wired to disk.
    console = Console(file=sys.stdout)
    case_rows = [
        {"name": case_name(case), "status": "pending", "done": 0,
         "total": None, "sr": None, "spl": None, "timeout": None}
        for case in cases
    ]

    with Live(
        render_dashboard(case_rows),
        console=console,
        refresh_per_second=4,
        redirect_stdout=False,
        redirect_stderr=False,
    ) as live:
        for row, case in zip(case_rows, cases):
            config = copy.deepcopy(base_config)
            if args.method is not None:
                config.method.name = args.method
            if args.dataset_path is not None:
                config.evaluation.dataset_path = args.dataset_path
            if args.max_episodes is not None:
                config.evaluation.max_episodes = args.max_episodes

            config.evaluation.query_type = START_TO_QUERY_TYPE[case["start"]]
            config.evaluation.tracking_mode = DIRECTION_TO_TRACKING_MODE[case["direction"]]
            config.evaluation.use_augmented = CAMERA_TO_AUGMENTED[case["camera"]]
            # The camera source follows the camera condition: the matched case
            # renders with the reference trajectory's camera parameters, the cross
            # case with the query's. Pinning these on would render the matched case
            # at the wrong sensor height / FOV / aspect ratio.
            config.evaluation.change_camera = config.evaluation.use_augmented
            config.evaluation.change_height = config.evaluation.use_augmented
            config.evaluation.log_dir = str(base_log_dir / case_name(case) / config.method.name)

            method_config = load_method_config(config)
            method_cls = get_method(config.method.name)
            method = method_cls(method_config)

            log_dir = Path(config.evaluation.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(config_path, log_dir / "eval_config.yaml")
            if config.method.config_path is not None:
                shutil.copy2(config.method.config_path, log_dir / "method_config.yaml")
            else:
                with open(log_dir / "method_config.yaml", "w") as f:
                    yaml.safe_dump(method_config, f, sort_keys=False)

            row["status"] = "running"
            live.update(render_dashboard(case_rows))

            def on_episode(done, total, summary, result, _row=row):
                _row["done"] = done
                _row["total"] = total
                _row["sr"] = summary["success_rate"]
                _row["spl"] = summary["mean_spl"]
                _row["timeout"] = summary["timeout_rate"]
                live.update(render_dashboard(case_rows))

            # All of the runner's verbose output (setup, per-episode, per-step,
            # timing breakdowns) is wired to <log_dir>/worker.log so the terminal
            # only shows the dashboard.
            worker_log = log_dir / "worker.log"
            from .runner import EvaluationRunner

            try:
                with open(worker_log, "w", buffering=1) as logf, \
                        contextlib.redirect_stdout(logf), \
                        contextlib.redirect_stderr(logf):
                    runner = EvaluationRunner(config, method)
                    row["total"] = len(runner.dataset)
                    live.update(render_dashboard(case_rows))
                    runner.run(start_idx=args.start_idx, on_episode=on_episode, visualize=args.visualize, materialize=args.materialize)
                row["status"] = "done"
            except Exception:
                row["status"] = "error"
                with open(worker_log, "a") as logf:
                    traceback.print_exc(file=logf)
            live.update(render_dashboard(case_rows))

    console.print(
        f"[green]Evaluation complete.[/green] Per-case results.json, worker.log, "
        f"and configs written under [cyan]{base_log_dir}[/cyan]"
    )


def render_dashboard(case_rows) -> Table:
    """Render the per-case evaluation dashboard table."""
    table = Table(box=box.SIMPLE_HEAVY, title="LoTIS navigation evaluation")
    table.add_column("Case", style="cyan", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Episodes", justify="right")
    table.add_column("Success", justify="right")
    table.add_column("SPL", justify="right")
    table.add_column("Timeout", justify="right")

    status_style = {"pending": "dim", "running": "yellow", "done": "green", "error": "red"}
    for r in case_rows:
        total = r["total"]
        episodes = f"{r['done']}/{total}" if total is not None else str(r["done"])
        sr = f"{r['sr']:.1%}" if r["sr"] is not None else "-"
        spl = f"{r['spl']:.3f}" if r["spl"] is not None else "-"
        timeout = f"{r['timeout']:.1%}" if r["timeout"] is not None else "-"
        style = status_style.get(r["status"], "")
        table.add_row(r["name"], f"[{style}]{r['status']}[/{style}]", episodes, sr, spl, timeout)
    return table


def expand_cases(starts=None, directions=None, cameras=None):
    starts = starts or ["on", "off"]
    directions = directions or ["forward", "backward", "any"]
    cameras = cameras or ["matched", "cross"]

    cases = []
    for start in starts:
        for direction in directions:
            if start == "on" and direction == "any":
                continue
            for camera in cameras:
                cases.append({"start": start, "direction": direction, "camera": camera})
    return cases


def case_name(case):
    return f"{case['start']}_start_{case['direction']}_{case['camera']}_camera"


if __name__ == "__main__":
    main()
