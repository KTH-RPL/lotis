#!/usr/bin/env python3
"""
LoTIS inference script with Rerun visualization.

Localizes a reference trajectory in a query image stream and visualizes
the predictions live in Rerun: the query view with trajectory points
overlaid, and the closest matching trajectory frame side by side.

Usage:
    # Static query — image file, video, or directory of images
    uv run python infer.py --trajectory path/to/traj.mp4 --query path/to/query.jpg

    # Live webcam
    uv run python infer.py --trajectory path/to/traj.mp4 --usb-cam
    uv run python infer.py --trajectory path/to/traj.mp4 --usb-cam --cam-id 1

    # Built-in example
    uv run python infer.py \
        --trajectory examples/00_KTH_Campus/Courtyard/trajectory.mp4 \
        --query examples/00_KTH_Campus/Courtyard/queries/Forward/query.mp4
"""

import argparse
import os
import sys

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
from PIL import Image

from lotis import TrajectoryLocalizer
from lotis.preprocessing import load_images_from_path

_COLORMAP = cv2.COLORMAP_JET


def _build_blueprint() -> rrb.Blueprint:
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial2DView(name="Query", origin="query"),
            rrb.Spatial2DView(name="Closest trajectory frame", origin="trajectory"),
        ),
        collapse_panels=True,
    )


def _log_result(
    frame_rgb: np.ndarray,
    result,
    traj_frames: list,
    vis_threshold: float,
) -> None:
    rr.log("query", rr.Image(frame_rgb).compress(jpeg_quality=90))

    vis_mask = result.visibility > vis_threshold
    vis_indices = np.where(vis_mask)[0]
    if vis_indices.size == 0:
        return

    vis_coords = result.coords[vis_mask]  # (row, col) in [-1, 1]
    h, w = frame_rgb.shape[:2]

    # Convert to pixel (x=col, y=row) for Rerun's 2D coordinate system
    px = (vis_coords[:, 1] + 1.0) / 2.0 * w
    py = (vis_coords[:, 0] + 1.0) / 2.0 * h
    positions = np.stack([px, py], axis=-1)

    # JET colormap: blue = trajectory start, red = trajectory end
    n_traj = len(result.visibility)
    cvals = (255 * vis_indices / max(1, n_traj - 1)).astype(np.uint8)
    colors_bgr = cv2.applyColorMap(cvals[:, None], _COLORMAP)[:, 0, :]
    colors_rgb = colors_bgr[:, ::-1]

    # Point size: larger = closer to camera
    base_r = max(h, w) * 0.015
    if result.distances is not None:
        d = result.distances[vis_mask]
        d_min, d_max = d.min(), d.max()
        norm = (d - d_min) / (d_max - d_min) if d_max > d_min else np.zeros_like(d)
        radii = base_r * (1.0 - 0.7 * norm)
    else:
        radii = np.full(len(vis_indices), base_r)

    rr.log("query/points", rr.Points2D(positions=positions, colors=colors_rgb, radii=radii))

    # Lines between consecutive visible trajectory points
    if len(positions) > 1:
        consecutive = vis_indices[1:] == vis_indices[:-1] + 1
        segments = np.stack([positions[:-1], positions[1:]], axis=1)[consecutive]
        seg_colors = colors_rgb[:-1][consecutive]
        rr.log("query/lines", rr.LineStrips2D(segments, colors=seg_colors))

    closest = result.closest_frame()
    rr.log("trajectory", rr.Image(np.asarray(traj_frames[closest])).compress(jpeg_quality=90))


def _run_static(localizer, encoding, traj_frames: list, query_path: str, args) -> None:
    frames = load_images_from_path(query_path)
    print(f"Loaded {len(frames)} query frame(s) from {query_path}")
    for i, frame in enumerate(frames):
        result = localizer.localize(frame, encoding)
        rr.set_time("frame", sequence=i)
        _log_result(np.asarray(frame), result, traj_frames, args.vis_threshold)
    print("Done.")


def _run_webcam(localizer, encoding, traj_frames: list, cam_id: int, args) -> None:
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print(f"Error: could not open camera {cam_id}.", file=sys.stderr)
        sys.exit(1)

    print(f"Streaming from camera {cam_id}. Press Ctrl-C to stop.")
    frame_idx = 0
    try:
        while True:
            ret, bgr = cap.read()
            if not ret:
                print("Failed to read frame.", file=sys.stderr)
                break
            frame_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            result = localizer.localize(Image.fromarray(frame_rgb), encoding)
            rr.set_time("frame", sequence=frame_idx)
            _log_result(frame_rgb, result, traj_frames, args.vis_threshold)
            frame_idx += 1
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
    print(f"Processed {frame_idx} frames.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Localize a reference trajectory in query images and visualize with Rerun.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--trajectory", required=True,
        help="Reference trajectory: video file, image directory, or single image.",
    )

    query = parser.add_mutually_exclusive_group(required=True)
    query.add_argument(
        "--query",
        help="Query input: video file, image directory, or single image.",
    )
    query.add_argument(
        "--usb-cam", action="store_true",
        help="Use a connected webcam as live query input.",
    )

    parser.add_argument("--cam-id", type=int, default=0, help="Camera device ID.")
    parser.add_argument(
        "--checkpoint", default=os.getenv("LOTIS_CHECKPOINT", "final_model.pth"),
        help="Path to LoTIS model checkpoint.",
    )
    parser.add_argument(
        "--config", default=os.getenv("LOTIS_CONFIG", "final_config.yaml"),
        help="Path to LoTIS model config.",
    )
    parser.add_argument(
        "--dinov3-weights", default=os.getenv("DINOV3_WEIGHTS"),
        help="Path to DINOv3 weights file. Falls back to DINOV3_WEIGHTS env var.",
    )
    parser.add_argument(
        "--dinov3-repo", default=os.getenv("DINOV3_REPO", "./dinov3"),
        help="Path to the cloned facebookresearch/dinov3 repository.",
    )
    parser.add_argument(
        "--device", default=None,
        help="Device: 'cuda' or 'cpu'. Auto-detects if not set.",
    )
    parser.add_argument(
        "--vis-threshold", type=float, default=0.5, metavar="T",
        help="Visibility threshold for displaying points.",
    )
    parser.add_argument(
        "--port", type=int, default=9876,
        help="Port for the spawned Rerun viewer (default: 9876).",
    )

    rr.script_add_args(parser)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading LoTIS model...")
    localizer = TrajectoryLocalizer.from_checkpoint(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=device,
        dinov3_weights=args.dinov3_weights,
        dinov3_repo=args.dinov3_repo,
    )

    print(f"Encoding trajectory: {args.trajectory}")
    traj_frames = load_images_from_path(args.trajectory, max_images=localizer.max_seq_len)
    encoding = localizer.encode_trajectory(traj_frames)
    print(f"Encoded {len(traj_frames)} trajectory frames.")

    rr.script_setup(args, "lotis")

    # Spawn a local viewer unless connecting to a remote or writing to a file.
    _remote = args.connect or args.serve or bool(getattr(args, "save", None)) or getattr(args, "stdout", False) or args.headless
    if not _remote:
        rr.spawn(port=args.port)

    rr.send_blueprint(_build_blueprint())

    if args.usb_cam:
        _run_webcam(localizer, encoding, traj_frames, args.cam_id, args)
    else:
        _run_static(localizer, encoding, traj_frames, args.query, args)

    rr.script_teardown(args)


if __name__ == "__main__":
    main()
