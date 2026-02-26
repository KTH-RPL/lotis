"""
LoTIS Demo - Trajectory Localization in Image Space

Gradio app for localizing query images within trajectory videos.
Works locally and on Hugging Face Spaces with ZeroGPU.
"""

import base64
import io
import json
import os
import re
import tempfile
import time

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image

from lotis import TrajectoryLocalizer

# ---------------------------------------------------------------------------
# ZeroGPU compatibility
# ---------------------------------------------------------------------------
try:
    import spaces
    ZERO_GPU = True
except ImportError:
    ZERO_GPU = False


def gpu_decorator(duration=120):
    def wrapper(fn):
        if ZERO_GPU:
            return spaces.GPU(duration=duration)(fn)
        return fn
    return wrapper


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
CHECKPOINT = os.environ.get("LOTIS_CHECKPOINT", "final_model.pth")
CONFIG = os.environ.get("LOTIS_CONFIG", "final_config.yaml")
FEATURE_EXTRACTOR = os.environ.get("LOTIS_FEATURE_EXTRACTOR", "dinov3")
DINOV3_WEIGHTS = os.environ.get("DINOV3_WEIGHTS", "")
DINOV3_REPO = os.environ.get("DINOV3_REPO", "./dinov3")

localizer: TrajectoryLocalizer | None = None


def get_localizer() -> TrajectoryLocalizer:
    global localizer
    if localizer is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        localizer = TrajectoryLocalizer.from_checkpoint(
            checkpoint_path=CHECKPOINT,
            config_path=CONFIG,
            device=device,
            feature_extractor_type=FEATURE_EXTRACTOR,
            dinov3_weights=DINOV3_WEIGHTS or None,
            dinov3_repo=DINOV3_REPO,
        )
    return localizer


# ---------------------------------------------------------------------------
# Video / image loading helpers
# ---------------------------------------------------------------------------

def get_video_fps(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return fps


def load_frames_from_video(
    video_path: str,
    max_frames: int | None = None,
    subsample: int = 1,
) -> list[Image.Image]:
    cap = cv2.VideoCapture(video_path)

    # Read sequentially, only decode every Nth frame (grab+retrieve is faster than seek)
    frames = []
    idx = 0
    while True:
        ret = cap.grab()
        if not ret:
            break
        if idx % subsample == 0:
            ret, frame = cap.retrieve()
            if ret:
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        idx += 1
    cap.release()

    if max_frames and max_frames > 0 and len(frames) > max_frames:
        step = int(np.ceil(len(frames) / max_frames))
        frames = frames[::step]
    return frames


def load_query_frames(
    query_video: str | None,
    query_images: list[str] | None,
    max_frames: int | None = None,
    subsample: int = 1,
) -> list[Image.Image]:
    """Load query frames from either a video path or a list of image paths."""
    if query_video:
        return load_frames_from_video(query_video, max_frames, subsample=subsample)
    if query_images:
        frames = []
        for path in sorted(query_images):
            img = Image.open(path).convert("RGB")
            frames.append(img)
        if max_frames and max_frames > 0 and len(frames) > max_frames:
            indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
            frames = [frames[i] for i in indices]
        return frames
    raise gr.Error("Please provide either a query video or query images.")


# ---------------------------------------------------------------------------
# Static visualization (for video output)
# ---------------------------------------------------------------------------
COLORMAP = cv2.COLORMAP_JET


def draw_predictions(
    query_frame: np.ndarray,
    result,
    vis_threshold: float = 0.5,
) -> np.ndarray:
    """Draw localization predictions on a query frame."""
    h, w = query_frame.shape[:2]
    canvas = query_frame.copy()

    visible_mask = result.visibility > vis_threshold
    if not visible_mask.any():
        return canvas

    visible_indices = np.where(visible_mask)[0]
    visible_coords = result.coords[visible_mask]
    n_total = len(result.visibility)

    # Coords are [row, col] in [-1, 1] -> pixel [x, y]
    pixel_xy = (visible_coords[:, ::-1] + 1.0) / 2.0 * np.array([w, h])

    # Colormap: position along trajectory (blue=start, red=end)
    color_values = (255 * visible_indices / max(1, n_total - 1)).astype(np.uint8)
    colors_bgr = cv2.applyColorMap(color_values, COLORMAP)[:, 0]  # [N, 3] BGR
    colors_rgb = colors_bgr[:, ::-1]  # canvas is RGB

    # Point radii based on distance (closer = bigger)
    base_radius = max(h, w) * 0.015
    if result.distances is not None:
        visible_dists = result.distances[visible_mask]
        d_min, d_max = visible_dists.min(), visible_dists.max()
        if d_max > d_min:
            norm = (visible_dists - d_min) / (d_max - d_min)
            radii = base_radius * (1.0 - 0.7 * norm)
        else:
            radii = np.full(len(visible_coords), base_radius)
    else:
        radii = np.full(len(visible_coords), base_radius)

    # Draw lines between consecutive visible points
    if len(pixel_xy) > 1:
        for i in range(len(pixel_xy) - 1):
            pt1 = tuple(pixel_xy[i].astype(int))
            pt2 = tuple(pixel_xy[i + 1].astype(int))
            color = tuple(int(c) for c in colors_rgb[i])
            cv2.line(canvas, pt1, pt2, color, thickness=max(1, int(base_radius * 0.5)))

    # Draw points on top of lines
    for i in range(len(pixel_xy)):
        center = tuple(pixel_xy[i].astype(int))
        color = tuple(int(c) for c in colors_rgb[i])
        r = max(2, int(radii[i]))
        cv2.circle(canvas, center, r, color, -1)
        cv2.circle(canvas, center, r, (255, 255, 255), max(1, r // 4))

    # Mark closest frame with a crosshair
    closest_idx = result.closest_frame()
    if visible_mask[closest_idx]:
        pos_in_visible = np.searchsorted(visible_indices, closest_idx)
        if pos_in_visible < len(pixel_xy):
            cx, cy = pixel_xy[pos_in_visible].astype(int)
            size = int(base_radius * 3)
            cv2.drawMarker(
                canvas, (cx, cy), (0, 255, 0),
                cv2.MARKER_CROSS, size, max(1, int(base_radius * 0.5)),
            )

    return canvas


def frames_to_video(frames: list[np.ndarray], fps: float = 10.0) -> str:
    """Encode annotated frames into an mp4 video, returned as a temp file path."""
    if not frames:
        return None
    h, w = frames[0].shape[:2]
    path = tempfile.mktemp(suffix=".mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    return path


# ---------------------------------------------------------------------------
# Interactive HTML/JS viewer
# ---------------------------------------------------------------------------
VIEWER_MAX_WIDTH = 800


def _frame_to_base64(frame: np.ndarray, max_w: int = VIEWER_MAX_WIDTH) -> tuple[str, int, int]:
    """Compress a frame to JPEG base64. Returns (data_url, width, height)."""
    h, w = frame.shape[:2]
    if w != max_w:
        scale = max_w / w
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        frame = cv2.resize(frame, (max_w, int(h * scale)), interpolation=interp)
        h, w = frame.shape[:2]
    # cv2.imencode is ~3x faster than PIL for JPEG
    _, buf = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
    b64 = base64.b64encode(buf.tobytes()).decode()
    return f"data:image/jpeg;base64,{b64}", w, h


def _jet_rgb(fraction: float) -> list[int]:
    """Map 0..1 through JET colormap, return [r, g, b]."""
    v = np.clip(np.array([[int(fraction * 255)]], dtype=np.uint8), 0, 255)
    bgr = cv2.applyColorMap(v, COLORMAP)[0, 0]
    return [int(bgr[2]), int(bgr[1]), int(bgr[0])]


def _thumbnail_base64(frame: np.ndarray, max_w: int = 120) -> str:
    """Encode a frame as a small JPEG thumbnail data URL."""
    h, w = frame.shape[:2]
    scale = max_w / w
    thumb = cv2.resize(frame, (max_w, int(h * scale)), interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", cv2.cvtColor(thumb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 70])
    b64 = base64.b64encode(buf.tobytes()).decode()
    return f"data:image/jpeg;base64,{b64}"


def build_interactive_viewer(
    frames: list[np.ndarray],
    results: list,
    fps: float,
    traj_frames: list[np.ndarray] | None = None,
) -> str:
    """Build a self-contained HTML/JS interactive annotation viewer."""
    # Encode frames
    t0 = time.time()
    frame_urls = []
    vw, vh = 0, 0
    for f in frames:
        url, vw, vh = _frame_to_base64(f)
        frame_urls.append(url)
    print(f"[viewer]   base64 encode {len(frames)} frames: {time.time() - t0:.3f}s")

    # Scale factor for coordinates (original frame -> viewer size)
    orig_h, orig_w = frames[0].shape[:2]
    sx = vw / orig_w
    sy = vh / orig_h

    # Pack prediction data per frame
    t0 = time.time()
    # Precompute colors once (same for all frames since n is always trajectory length)
    n_traj = len(results[0].visibility)
    color_indices = np.linspace(0, 255, n_traj, dtype=np.uint8)
    colors_bgr = cv2.applyColorMap(color_indices, COLORMAP)[:, 0]
    colors_list = [[int(c[2]), int(c[1]), int(c[0])] for c in colors_bgr]

    pred_data = []
    for result in results:
        n = len(result.visibility)
        # Coords [row, col] in [-1,1] -> pixel [x, y] in viewer space
        px = ((result.coords[:, 1] + 1.0) / 2.0 * orig_w * sx).tolist()
        py = ((result.coords[:, 0] + 1.0) / 2.0 * orig_h * sy).tolist()
        vis = result.visibility.tolist()
        dists = result.distances.tolist() if result.distances is not None else None
        closest = int(result.closest_frame())
        pred_data.append({
            "px": px, "py": py, "vis": vis, "dists": dists,
            "closest": closest, "colors": colors_list, "n": n,
        })
    print(f"[viewer]   pack pred data: {time.time() - t0:.3f}s")

    # Encode trajectory thumbnails for tooltip display
    t0 = time.time()
    traj_thumbs = []
    if traj_frames is not None:
        for tf in traj_frames:
            traj_thumbs.append(_thumbnail_base64(tf))
    print(f"[viewer]   traj thumbnails ({len(traj_thumbs)}): {time.time() - t0:.3f}s")

    t0 = time.time()
    data_json = json.dumps({"frames": frame_urls, "preds": pred_data, "fps": fps, "w": vw, "h": vh, "trajThumbs": traj_thumbs})
    print(f"[viewer]   json.dumps ({len(data_json)//1024}KB): {time.time() - t0:.3f}s")

    t0 = time.time()
    viewer_doc = _VIEWER_HTML.replace("__DATA__", data_json)
    import html as html_mod
    escaped = html_mod.escape(viewer_doc)
    result_html = f'<div style="display:flex;justify-content:center"><iframe srcdoc="{escaped}" style="width:{vw}px;height:{vh + 120}px;border:none"></iframe></div>'
    print(f"[viewer]   html escape + build ({len(result_html)//1024}KB): {time.time() - t0:.3f}s")
    return result_html


_VIEWER_HTML = r"""
<div id="lotis-root" style="font-family:system-ui,sans-serif;max-width:__MAX_W__px">
  <div style="position:relative;display:inline-block;line-height:0">
    <canvas id="lotis-bg"></canvas>
    <canvas id="lotis-overlay" style="position:absolute;top:0;left:0;cursor:crosshair"></canvas>
    <div id="lotis-tooltip" style="
      display:none;position:absolute;pointer-events:none;
      background:rgba(0,0,0,0.85);color:#fff;padding:8px 12px;border-radius:4px;
      font-size:13px;line-height:1.6;white-space:nowrap;z-index:10;
    "></div>
  </div>

  <div style="display:flex;align-items:center;gap:8px;margin:8px 0 4px 0">
    <button id="lotis-play" style="
      width:32px;height:32px;border:none;border-radius:4px;cursor:pointer;
      background:#4a90d9;color:#fff;font-size:16px;display:flex;align-items:center;justify-content:center;
    ">&#9654;</button>
    <input id="lotis-slider" type="range" min="0" max="0" value="0" step="1"
      style="flex:1;cursor:pointer">
    <span id="lotis-counter" style="font-size:13px;min-width:60px;text-align:right;color:#555"></span>
    <select id="lotis-speed" style="
      font-size:12px;padding:2px 4px;border-radius:3px;border:1px solid #ccc;cursor:pointer;
    ">
      <option value="0.25">0.25x</option>
      <option value="0.5">0.5x</option>
      <option value="1" selected>1x</option>
      <option value="2">2x</option>
      <option value="4">4x</option>
    </select>
  </div>

  <div style="display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin:4px 0;font-size:13px;color:#444">
    <label style="display:flex;align-items:center;gap:4px">
      <input type="checkbox" id="lotis-lines" checked> Path
    </label>
    <label style="display:flex;align-items:center;gap:4px">
      <input type="checkbox" id="lotis-points" checked> Points
    </label>
    <label style="display:flex;align-items:center;gap:4px">
      <input type="checkbox" id="lotis-closest"> Closest
    </label>
    <label style="display:flex;align-items:center;gap:4px">
      Threshold
      <input type="range" id="lotis-thresh" min="0" max="1" step="0.05" value="0.5"
        style="width:100px">
      <span id="lotis-thresh-val">0.50</span>
    </label>
  </div>
</div>

<script>
(function() {
  const D = __DATA__;
  const frames = D.frames, preds = D.preds, fps = D.fps, W = D.w, H = D.h;
  const N = frames.length;

  const bgCanvas = document.getElementById('lotis-bg');
  const overlay = document.getElementById('lotis-overlay');
  const tooltip = document.getElementById('lotis-tooltip');
  const slider = document.getElementById('lotis-slider');
  const counter = document.getElementById('lotis-counter');
  const playBtn = document.getElementById('lotis-play');
  const chkLines = document.getElementById('lotis-lines');
  const chkPoints = document.getElementById('lotis-points');
  const chkClosest = document.getElementById('lotis-closest');
  const threshSlider = document.getElementById('lotis-thresh');
  const threshVal = document.getElementById('lotis-thresh-val');
  const speedSelect = document.getElementById('lotis-speed');

  bgCanvas.width = overlay.width = W;
  bgCanvas.height = overlay.height = H;
  const bgCtx = bgCanvas.getContext('2d');
  const ctx = overlay.getContext('2d');

  slider.max = N - 1;
  let curFrame = 0;
  let playing = false;
  let animId = null;
  let lastTime = 0;

  // Preload images
  const imgs = [];
  let loaded = 0;
  frames.forEach((src, i) => {
    const img = new window.Image();
    img.onload = () => { loaded++; if (i === 0) draw(); };
    img.src = src;
    imgs.push(img);
  });

  function getThreshold() { return parseFloat(threshSlider.value); }

  function draw() {
    // Background frame
    if (imgs[curFrame] && imgs[curFrame].complete) {
      bgCtx.drawImage(imgs[curFrame], 0, 0, W, H);
    }
    // Overlay annotations
    ctx.clearRect(0, 0, W, H);
    const p = preds[curFrame];
    if (!p) return;
    const thresh = getThreshold();
    const showLines = chkLines.checked;
    const showPoints = chkPoints.checked;
    const showClosest = chkClosest.checked;

    // Collect visible points
    const vis = [];
    for (let i = 0; i < p.n; i++) {
      if (p.vis[i] > thresh) vis.push(i);
    }

    const baseR = Math.max(W, H) * 0.012;

    // Radii from distance
    const radii = new Array(p.n).fill(baseR);
    if (p.dists) {
      let dMin = Infinity, dMax = -Infinity;
      for (const i of vis) { dMin = Math.min(dMin, p.dists[i]); dMax = Math.max(dMax, p.dists[i]); }
      if (dMax > dMin) {
        for (const i of vis) {
          const norm = (p.dists[i] - dMin) / (dMax - dMin);
          radii[i] = baseR * (1.0 - 0.7 * norm);
        }
      }
    }

    // Lines (only between consecutive trajectory indices)
    if (showLines && vis.length > 1) {
      ctx.lineWidth = Math.max(1, baseR * 0.4);
      for (let j = 0; j < vis.length - 1; j++) {
        const a = vis[j], b = vis[j+1];
        if (b !== a + 1) continue;
        const [r,g,bl] = p.colors[a];
        ctx.strokeStyle = `rgba(${r},${g},${bl},0.7)`;
        ctx.beginPath();
        ctx.moveTo(p.px[a], p.py[a]);
        ctx.lineTo(p.px[b], p.py[b]);
        ctx.stroke();
      }
    }

    // Points
    if (showPoints) {
      for (const i of vis) {
        const [r,g,b] = p.colors[i];
        const rad = Math.max(2, radii[i]);
        ctx.beginPath();
        ctx.arc(p.px[i], p.py[i], rad, 0, Math.PI * 2);
        ctx.fillStyle = `rgb(${r},${g},${b})`;
        ctx.fill();
        ctx.strokeStyle = 'rgba(255,255,255,0.8)';
        ctx.lineWidth = Math.max(1, rad * 0.25);
        ctx.stroke();
      }
    }

    // Closest marker
    if (showClosest && p.vis[p.closest] > thresh) {
      const cx = p.px[p.closest], cy = p.py[p.closest];
      const s = baseR * 2.5;
      ctx.strokeStyle = '#00ff00';
      ctx.lineWidth = Math.max(2, baseR * 0.4);
      ctx.beginPath(); ctx.moveTo(cx - s, cy); ctx.lineTo(cx + s, cy); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(cx, cy - s); ctx.lineTo(cx, cy + s); ctx.stroke();
    }

    counter.textContent = `${curFrame + 1} / ${N}`;
  }

  function setFrame(i) {
    curFrame = Math.max(0, Math.min(N - 1, i));
    slider.value = curFrame;
    draw();
  }

  // Playback
  function getSpeed() { return parseFloat(speedSelect.value); }

  function tick(ts) {
    if (!playing) return;
    if (ts - lastTime >= 1000 / (fps * getSpeed())) {
      lastTime = ts;
      if (curFrame < N - 1) { setFrame(curFrame + 1); }
      else { playing = false; playBtn.innerHTML = '&#9654;'; return; }
    }
    animId = requestAnimationFrame(tick);
  }

  playBtn.addEventListener('click', () => {
    playing = !playing;
    playBtn.innerHTML = playing ? '&#9646;&#9646;' : '&#9654;';
    if (playing) {
      if (curFrame >= N - 1) curFrame = 0;
      lastTime = performance.now();
      animId = requestAnimationFrame(tick);
    }
  });

  slider.addEventListener('input', () => setFrame(parseInt(slider.value)));

  threshSlider.addEventListener('input', () => {
    threshVal.textContent = parseFloat(threshSlider.value).toFixed(2);
    draw();
  });
  chkLines.addEventListener('change', draw);
  chkPoints.addEventListener('change', draw);
  chkClosest.addEventListener('change', draw);

  // Hover detection
  overlay.addEventListener('mousemove', (e) => {
    const rect = overlay.getBoundingClientRect();
    const mx = (e.clientX - rect.left) * (W / rect.width);
    const my = (e.clientY - rect.top) * (H / rect.height);
    const p = preds[curFrame];
    const thresh = getThreshold();
    let hit = -1, bestDist = 15 * 15;  // 15px hit radius

    for (let i = 0; i < p.n; i++) {
      if (p.vis[i] <= thresh) continue;
      const dx = p.px[i] - mx, dy = p.py[i] - my;
      const d2 = dx*dx + dy*dy;
      if (d2 < bestDist) { bestDist = d2; hit = i; }
    }

    if (hit >= 0) {
      let html = '';
      if (D.trajThumbs && D.trajThumbs[hit]) {
        html += `<img src="${D.trajThumbs[hit]}" style="display:block;border-radius:3px;margin-bottom:6px;max-width:120px">`;
      }
      html += `<b>Traj frame ${hit}</b><br>Visibility: ${p.vis[hit].toFixed(3)}`;
      if (p.dists) html += `<br>Distance: ${p.dists[hit].toFixed(3)}`;
      if (hit === p.closest) html += `<br><span style="color:#0f0">&#10010; Closest</span>`;
      tooltip.innerHTML = html;
      tooltip.style.display = 'block';
      // Position tooltip near cursor but within bounds
      const tx = Math.min(e.clientX - rect.left + 12, rect.width - 160);
      const ty = Math.max(e.clientY - rect.top - 130, 0);
      tooltip.style.left = tx + 'px';
      tooltip.style.top = ty + 'px';
    } else {
      tooltip.style.display = 'none';
    }
  });

  overlay.addEventListener('mouseleave', () => { tooltip.style.display = 'none'; });

  draw();
})();
</script>
""".replace("__MAX_W__", str(VIEWER_MAX_WIDTH))


# ---------------------------------------------------------------------------
# Main prediction pipeline
# ---------------------------------------------------------------------------

MAX_TRAJ_FRAMES = 40
QUERY_BATCH_SIZE = 64


@gpu_decorator(duration=120)
def predict(
    trajectory_video: str | None,
    trajectory_images: list[str] | None,
    query_video: str | None,
    query_images: list[str] | None,
    max_query_frames: int,
    vis_threshold: float,
    query_subsample: int,
):
    t_total = time.time()

    # --- Model setup ---
    t0 = time.time()
    loc = get_localizer()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loc.model.to(device)
    loc.feature_extractor.to(device)
    loc.device = device
    loc.use_amp = device == "cuda"
    print(f"[timing] Model setup: {time.time() - t0:.3f}s")

    # --- Load trajectory frames (always 40) ---
    t0 = time.time()
    if trajectory_video:
        traj_frames = load_frames_from_video(trajectory_video, max_frames=MAX_TRAJ_FRAMES)
    elif trajectory_images:
        traj_frames = []
        for path in sorted(trajectory_images):
            img = Image.open(path).convert("RGB")
            traj_frames.append(img)
        if len(traj_frames) > MAX_TRAJ_FRAMES:
            indices = np.linspace(0, len(traj_frames) - 1, MAX_TRAJ_FRAMES, dtype=int)
            traj_frames = [traj_frames[i] for i in indices]
    else:
        raise gr.Error("Please provide a trajectory video or trajectory images.")

    if len(traj_frames) < 2:
        raise gr.Error("Trajectory must contain at least 2 frames.")
    print(f"[timing] Load trajectory ({len(traj_frames)} frames): {time.time() - t0:.3f}s")

    # --- Load query frames ---
    t0 = time.time()
    max_q = max_query_frames if max_query_frames > 0 else None
    sub = int(query_subsample) if query_subsample and query_subsample > 1 else 1
    query_frames = load_query_frames(query_video, query_images, max_frames=max_q, subsample=sub)
    if not query_frames:
        raise gr.Error("No query frames found.")
    print(f"[timing] Load queries ({len(query_frames)} frames): {time.time() - t0:.3f}s")

    # --- Encode trajectory ---
    t0 = time.time()
    encoding = loc.encode_trajectory(traj_frames, max_frames=MAX_TRAJ_FRAMES)
    print(f"[timing] Trajectory encoding: {time.time() - t0:.3f}s")

    # --- Localize queries in batches ---
    t0 = time.time()
    n_batches = (len(query_frames) + QUERY_BATCH_SIZE - 1) // QUERY_BATCH_SIZE
    results = []
    for i in range(0, len(query_frames), QUERY_BATCH_SIZE):
        batch = query_frames[i : i + QUERY_BATCH_SIZE]
        tb = time.time()
        batch_results = loc.localize(batch, encoding)
        elapsed = time.time() - tb
        if not isinstance(batch_results, list):
            batch_results = [batch_results]
        results.extend(batch_results)
        batch_num = i // QUERY_BATCH_SIZE + 1
        print(f"[timing]   Query batch {batch_num}/{n_batches} ({len(batch)} frames): {elapsed:.3f}s ({elapsed/len(batch)*1000:.1f}ms/frame)")
    print(f"[timing] Query localization total: {time.time() - t0:.3f}s")

    output_fps = 30.0

    # --- Interactive viewer ---
    t0 = time.time()
    query_frames_np = [np.asarray(f) for f in query_frames]
    traj_frames_np = [np.asarray(f) for f in traj_frames]
    viewer_html = build_interactive_viewer(query_frames_np, results, output_fps, traj_frames=traj_frames_np)
    print(f"[timing] Interactive viewer build: {time.time() - t0:.3f}s")

    print(f"[timing] TOTAL: {time.time() - t_total:.3f}s")
    return viewer_html


# ---------------------------------------------------------------------------
# Example discovery
# ---------------------------------------------------------------------------
EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "examples")

# Expected layout:
#   examples/
#     <trajectory_name>/
#       trajectory.mp4          (or trajectory/ dir with images)
#       queries/
#         <category_label>/
#           query.mp4           (or image files directly)
#
# Each trajectory folder must contain either:
#   - trajectory.mp4   (video file)
#   - trajectory/      (directory of images, sorted alphabetically)
#
# Each query category folder must contain either:
#   - query.mp4        (video file)
#   - image files directly (*.jpg, *.png, etc.)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _find_video(directory: str) -> str | None:
    """Find the first video file in a directory (ignoring preview files)."""
    for f in sorted(os.listdir(directory)):
        if "_preview" in f:
            continue
        if os.path.splitext(f)[1].lower() in _VIDEO_EXTS:
            return os.path.join(directory, f)
    return None


def _find_images(directory: str) -> list[str]:
    """Find all image files in a directory, sorted."""
    imgs = []
    for f in sorted(os.listdir(directory)):
        if os.path.splitext(f)[1].lower() in _IMAGE_EXTS:
            imgs.append(os.path.join(directory, f))
    return imgs


def scan_examples() -> dict:
    """Scan the examples directory and return structured example data.

    Layout:
        examples/<group>/<trajectory>/trajectory.mp4
        examples/<group>/<trajectory>/queries/<category>/query.mp4

    Returns dict: {
        "group_name": {
            "trajectory_name": {
                "traj_video": str|None, "traj_images": list[str],
                "queries": {"category": {"video": str|None, "images": list[str]}}
            }
        }
    }
    """
    groups = {}
    if not os.path.isdir(EXAMPLES_DIR):
        return groups

    for group_dirname in sorted(os.listdir(EXAMPLES_DIR)):
        group_dir = os.path.join(EXAMPLES_DIR, group_dirname)
        if not os.path.isdir(group_dir):
            continue
        # Strip leading "NN_" ordering prefix, underscores to spaces
        group_name = re.sub(r"^\d+_", "", group_dirname).replace("_", " ")

        trajectories = {}
        for traj_name in sorted(os.listdir(group_dir)):
            traj_dir = os.path.join(group_dir, traj_name)
            if not os.path.isdir(traj_dir):
                continue

            traj_video = None
            traj_images = []
            traj_sub = os.path.join(traj_dir, "trajectory")
            if os.path.isdir(traj_sub):
                traj_video = _find_video(traj_sub)
                if not traj_video:
                    traj_images = _find_images(traj_sub)
            else:
                for name in ("trajectory.mp4", "trajectory.mov", "trajectory.avi"):
                    p = os.path.join(traj_dir, name)
                    if os.path.isfile(p):
                        traj_video = p
                        break

            if not traj_video and not traj_images:
                continue

            queries_dir = os.path.join(traj_dir, "queries")
            queries = {}
            if os.path.isdir(queries_dir):
                for cat_name in sorted(os.listdir(queries_dir)):
                    cat_dir = os.path.join(queries_dir, cat_name)
                    if not os.path.isdir(cat_dir):
                        continue
                    q_video = _find_video(cat_dir)
                    q_images = [] if q_video else _find_images(cat_dir)
                    if q_video or q_images:
                        queries[cat_name] = {"video": q_video, "images": q_images}

            if queries:
                trajectories[traj_name] = {
                    "traj_video": traj_video,
                    "traj_images": traj_images,
                    "queries": queries,
                }

        if trajectories:
            ref_path = os.path.join(group_dir, "reference.txt")
            reference = open(ref_path).read().strip() if os.path.isfile(ref_path) else None
            groups[group_name] = {"trajectories": trajectories, "reference": reference}

    return groups


def _get_thumbnail(traj_dir: str) -> str | None:
    """Get or generate a thumbnail for a trajectory."""
    thumb = os.path.join(traj_dir, "thumbnail.jpg")
    if os.path.isfile(thumb):
        return thumb
    # Try to extract from trajectory video
    vid = os.path.join(traj_dir, "trajectory.mp4")
    if os.path.isfile(vid):
        cap = cv2.VideoCapture(vid)
        ret, frame = cap.read()
        cap.release()
        if ret:
            cv2.imwrite(thumb, frame)
            return thumb
    # Try first image in trajectory/ subdir
    traj_sub = os.path.join(traj_dir, "trajectory")
    if os.path.isdir(traj_sub):
        imgs = _find_images(traj_sub)
        if imgs:
            import shutil
            shutil.copy2(imgs[0], thumb)
            return thumb
    return None


def _get_or_create_preview(video_path: str, max_width: int = 320) -> str | None:
    """Get or create a small low-res preview video for table display."""
    if not video_path:
        return None
    d, fname = os.path.split(video_path)
    name, ext = os.path.splitext(fname)
    preview_path = os.path.join(d, f"{name}_preview.mp4")
    if os.path.isfile(preview_path):
        return preview_path

    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if w <= max_width:
        cap.release()
        return video_path

    scale = max_width / w
    new_w = max_width
    new_h = int(h * scale) // 2 * 2  # ensure even

    # Take every 3rd frame, cap at 60 frames
    step = max(1, total // 60) if total > 60 else 1
    step = max(step, 3)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(preview_path, fourcc, 15.0, (new_w, new_h))
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            small = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            writer.write(small)
        idx += 1
    cap.release()
    writer.release()
    return preview_path


def _image_to_preview_video(image_path: str, max_width: int = 320) -> str | None:
    """Create a short single-frame video from an image for table display."""
    d, fname = os.path.split(image_path)
    name, _ = os.path.splitext(fname)
    preview_path = os.path.join(d, f"{name}_preview.mp4")
    if os.path.isfile(preview_path):
        return preview_path
    img = cv2.imread(image_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / w
        w, h = max_width, int(h * scale) // 2 * 2
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(preview_path, fourcc, 1.0, (w, h))
    for _ in range(3):
        writer.write(img)
    writer.release()
    return preview_path


def _query_preview(q: dict) -> str | None:
    """Get a video preview for the table — converts images to short videos."""
    if q["video"]:
        return _get_or_create_preview(q["video"])
    if q["images"]:
        return _image_to_preview_video(q["images"][0])
    return None


def build_examples_tables() -> dict[str, dict]:
    """Build per-group tables for gr.Examples with tabs.

    Returns: {"group_name": {"rows": [...], "reference": str|None}}
    """
    groups = scan_examples()
    tables = {}
    for group_name, group_data in groups.items():
        rows = []
        for traj_name, data in group_data["trajectories"].items():
            traj_preview = _get_or_create_preview(data["traj_video"])
            for cat_name, q in data["queries"].items():
                rows.append([
                    traj_name.replace("_", " "),
                    cat_name.replace("_", " "),
                    traj_preview,
                    _query_preview(q),
                ])
        if rows:
            display_name = group_name.replace("_", " ")
            tables[display_name] = {"rows": rows, "reference": group_data["reference"]}
    return tables


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

DESCRIPTION = """\
# LoTIS - Learning to Localize Trajectories in Image-Space for Visual Navigation

<div style="display:flex;gap:8px;margin-bottom:12px">
<a href="https://finnbusch.com/lotis/" target="_blank" style="text-decoration:none">
  <img src="https://img.shields.io/badge/Project_Page-blue?style=for-the-badge&logo=googlechrome&logoColor=white" alt="Project Page">
</a>
<a href="https://arxiv.org/abs/2602.18803" target="_blank" style="text-decoration:none">
  <img src="https://img.shields.io/badge/arXiv-Paper-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv">
</a>
</div>

Upload a **reference trajectory** and one or more **query** frames.
The model predicts where each camera pose of the reference trajectory appears in each query image,
visualized as colored points overlaid on each query frame.

- **Blue** = early in trajectory
- **Red** = late in trajectory
- Point size reflects predicted distance (larger = closer)
- Hover over points to see details and trajectory thumbnails.

<a id="scroll-to-examples" style="font-size:1.05em;cursor:pointer;color:#4a90d9">&#8595; Try the examples below</a>
"""

_EXAMPLES_TABLES = build_examples_tables()
if os.path.isdir(EXAMPLES_DIR):
    gr.set_static_paths(paths=[EXAMPLES_DIR])

_VIDEO_POPUP_HEAD = """
<style>
  button[role="tab"] { font-size: 1.25rem !important; padding: 12px 24px !important; }
  td.video video { pointer-events: none !important; }
  td.video .overlay { display: none !important; }
  td.video { cursor: pointer; }
  #video-popup {
    display:none; position:fixed; z-index:9999;
    pointer-events:none;
  }
  #video-popup.active { display:block; }
  #video-popup video {
    width:320px; border-radius:6px;
    box-shadow:0 4px 16px rgba(0,0,0,0.4);
    pointer-events:none;
  }
</style>
<script>
(function() {
    var ol = document.createElement('div');
    ol.id = 'video-popup';
    var vp = document.createElement('video');
    vp.id = 'video-popup-player';
    vp.controls = true; vp.autoplay = true; vp.loop = true;
    ol.appendChild(vp);
    document.body.appendChild(ol);

    var activeTd = null;

    document.addEventListener('mouseover', function(e) {
        var td = e.target.closest('td.video');
        if (td && td !== activeTd) {
            activeTd = td;
            var vid = td.querySelector('video');
            if (!vid) return;
            var src = vid.src || (vid.querySelector('source') && vid.querySelector('source').src);
            if (!src) return;
            vp.src = src;
            ol.classList.add('active');
        } else if (!td && activeTd) {
            activeTd = null;
            ol.classList.remove('active');
            vp.pause();
            vp.removeAttribute('src');
        }
    }, true);

    document.addEventListener('mousemove', function(e) {
        if (!activeTd) return;
        ol.style.left = (e.clientX - 160) + 'px';
        ol.style.top = (e.clientY - ol.offsetHeight - 16) + 'px';
    }, true);

    // "Try examples below" scroll link
    document.addEventListener('click', function(e) {
        if (e.target.id === 'scroll-to-examples' || e.target.closest('#scroll-to-examples')) {
            var el = document.getElementById('examples-section');
            if (el) el.scrollIntoView({behavior: 'smooth'});
        }
        // Scroll to top when clicking an example row (anywhere below the fold)
        if (e.target.closest('tr, .gallery-item')) {
            setTimeout(function() {
                document.body.scrollIntoView({behavior: 'smooth'});
            }, 300);
        }
    }, true);
})();
</script>
"""

with gr.Blocks() as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row(equal_height=False):
        # --- Left column: inputs stacked ---
        with gr.Column(scale=1):
            gr.Markdown("### Reference Trajectory")
            with gr.Tab("Video") as traj_vid_tab:
                traj_video = gr.Video(label="Trajectory video", height=240)
            with gr.Tab("Images") as traj_img_tab:
                traj_images = gr.File(
                    label="Trajectory frames",
                    file_count="multiple",
                    file_types=["image"],
                )
            # Clear the other input when switching tabs
            traj_vid_tab.select(fn=lambda: None, inputs=[], outputs=[traj_images])
            traj_img_tab.select(fn=lambda: None, inputs=[], outputs=[traj_video])

            gr.Markdown("### Query")
            with gr.Tabs() as query_tabs:
                with gr.Tab("Video", id="q_vid") as query_vid_tab:
                    query_video = gr.Video(label="Query video", height=240)
                with gr.Tab("Images", id="q_img") as query_img_tab:
                    query_images = gr.File(
                        label="Query frames",
                        file_count="multiple",
                        file_types=["image"],
                    )
            query_vid_tab.select(fn=lambda: None, inputs=[], outputs=[query_images])
            query_img_tab.select(fn=lambda: None, inputs=[], outputs=[query_video])

            query_subsample = gr.Slider(
                1, 30, value=5, step=1,
                label="Query subsample factor",
                info="Take every Nth frame (e.g. 5 = every 5th frame).",
            )
            max_query_frames = gr.Slider(value=0, visible=False)
            vis_threshold = gr.Slider(value=0.5, visible=False)

            run_btn = gr.Button("Run", variant="primary", size="lg")
            gr.DeepLinkButton()

        # --- Right column: results ---
        with gr.Column(scale=2):
            gr.Markdown("### Results\nHover over predicted points to see the corresponding trajectory frame.")
            viewer = gr.HTML(value="<div style='color:#888;padding:40px;text-align:center'>Run the model to see results</div>")

    # --- Wire run button ---
    run_btn.click(
        fn=predict,
        inputs=[
            traj_video, traj_images,
            query_video, query_images,
            max_query_frames, vis_threshold,
            query_subsample,
        ],
        outputs=[viewer],
    )

    # --- Examples table at bottom ---
    if _EXAMPLES_TABLES:
        gr.Markdown("### Examples\nClick a row to load it, then press **Run**.", elem_id="examples-section")

        ex_traj_name = gr.Textbox(visible=False, label="Name")
        ex_query_cat = gr.Textbox(visible=False, label="Characteristics")
        ex_traj_vid = gr.Video(visible=False, label="Trajectory")
        ex_query_vid = gr.Video(visible=False, label="Query")

        with gr.Tabs():
            for group_name, group_data in _EXAMPLES_TABLES.items():
                with gr.Tab(group_name):
                    gr.Examples(
                        examples=group_data["rows"],
                        inputs=[ex_traj_name, ex_query_cat, ex_traj_vid, ex_query_vid],
                        examples_per_page=20,
                    )
                    if group_data["reference"]:
                        gr.Markdown(f"*{group_data['reference']}*")

        def _load_example_videos(traj_name, query_cat):
            groups = scan_examples()
            for group_data in groups.values():
                for name, data in group_data["trajectories"].items():
                    if name.replace("_", " ") == traj_name:
                        for cat, q in data["queries"].items():
                            if cat.replace("_", " ") == query_cat:
                                tab = gr.Tabs(selected="q_vid" if q["video"] else "q_img")
                                return (
                                    data["traj_video"],
                                    q["video"],
                                    q["images"] or None,
                                    tab,
                                )
            raise gr.Error(f"Example not found: {traj_name} / {query_cat}")

        ex_query_cat.change(
            fn=_load_example_videos,
            inputs=[ex_traj_name, ex_query_cat],
            outputs=[traj_video, query_video, query_images, query_tabs],
        )

if __name__ == "__main__":
    demo.launch(allowed_paths=[EXAMPLES_DIR], theme=gr.themes.Soft(), ssr_mode=False, head=_VIDEO_POPUP_HEAD)
