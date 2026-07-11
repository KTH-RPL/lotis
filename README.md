# LoTIS: Learning to Localize Reference Trajectories in Image-Space for Visual Navigation

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2602.18803-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2602.18803)
[![Project Page](https://img.shields.io/badge/Project_Page-blue?style=for-the-badge&logo=googlechrome&logoColor=white)](https://finnbusch.com/lotis)
[![HuggingFace](https://img.shields.io/badge/Demo-HuggingFace-yellow?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/spaces/fnnBsch/lotis-demo)

*Finn Busch, Matti Vahs, Quantao Yang, Jesús Gerardo Ortega Peimbert, Yixi Cai, Jana Tumova, Olov Andersson*

*Division of Robotics, Perception, and Learning at KTH Royal Institute of Technology*

</div>

LoTIS is a model for visual navigation that provides robot-agnostic image-space guidance by localizing a reference RGB trajectory in the robot's current view. Given a reference trajectory (a sequence of RGB images) and a query image from the robot's current viewpoint, LoTIS predicts the 2D image-space coordinates, visibility, and relative distance of each trajectory pose as it would appear in the query view.

> **🆕 New — Navigation evaluation.** Reproduce the paper's Habitat benchmark and evaluate your own method against LoTIS. See [Navigation Evaluation](#navigation-evaluation).

## Setup

**1. Clone this repository and install dependencies**

```bash
git clone https://github.com/KTH-RPL/lotis.git
cd lotis
```

Dependencies are managed with [uv](https://docs.astral.sh/uv/). `uv run` will automatically install everything on first use. To install manually:

```bash
uv sync
```

**2. Clone the DINOv3 repository**

LoTIS uses [DINOv3](https://github.com/facebookresearch/dinov3) as a frozen backbone. Clone Meta's repository into the project root:

```bash
git clone https://github.com/facebookresearch/dinov3.git
```

**3. Download DINOv3 weights**

Request access to DINOv3 pretrained weights at [ai.meta.com/resources/models-and-libraries/dinov3-downloads](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/). Once approved, you will receive an email with download URLs. Download the ViT-B/16 pretrain weights (`dinov3_vitb16_pretrain.pth`) and set the path:

```bash
export DINOV3_WEIGHTS=/path/to/dinov3_vitb16_pretrain.pth
```

Or pass it directly via `--dinov3-weights` when running scripts.

**4. Download LoTIS model weights**

```bash
huggingface-cli download fnnBsch/lotis final_model.pth final_config.yaml --local-dir .
```

## Running inference

Localize a reference trajectory in a query (image, video, or directory), visualized live in [Rerun](https://rerun.io):

```bash
uv run python infer.py \
    --trajectory examples/00_KTH_Campus/Courtyard/trajectory.mp4 \
    --query examples/00_KTH_Campus/Courtyard/queries/Forward/query.mp4
```

**Live webcam** — encode a trajectory once, then localize from your camera in real time:

```bash
uv run python infer.py \
    --trajectory path/to/trajectory.mp4 \
    --usb-cam
```

Use `--cam-id 1` to select a different camera. All `infer.py` options:

```
--trajectory PATH       Reference trajectory (video, image dir, or single image)
--query PATH            Query input (video, image dir, or single image)
--usb-cam               Use webcam as live query  [mutually exclusive with --query]
--cam-id INT            Camera device ID (default: 0)
--checkpoint PATH       LoTIS checkpoint (default: final_model.pth)
--config PATH           LoTIS config (default: final_config.yaml)
--dinov3-weights PATH   DINOv3 weights file (or set DINOV3_WEIGHTS)
--dinov3-repo PATH      Path to cloned dinov3 repo (default: ./dinov3)
--device cuda|cpu       Inference device (auto-detected if not set)
--vis-threshold FLOAT   Visibility threshold for displaying points (default: 0.5)
```

Rerun also accepts additional flags (e.g. `--serve` to stream to a remote viewer). Run `python infer.py --help` for the full list.

### Remote visualization

If running on a remote machine, start the Rerun viewer locally and forward the port:

```bash
# On your local machine
rerun

# Forward the Rerun port from remote to local
ssh -R 9876:localhost:9876 <remote-host>

# On the remote machine
uv run python infer.py --trajectory ... --query ... --connect
```

## Gradio demo

A full interactive demo is available at the [HuggingFace Space](https://huggingface.co/spaces/fnnBsch/lotis-demo). To run it locally:

```bash
uv run python app.py
```

Set `DINOV3_WEIGHTS` (and optionally `DINOV3_REPO`) before running.

## Python API

```python
from lotis import TrajectoryEncoding, TrajectoryLocalizer

localizer = TrajectoryLocalizer.from_checkpoint(
    checkpoint_path="final_model.pth",
    config_path="final_config.yaml",
    dinov3_weights="/path/to/dinov3_vitb16_pretrain.pth",
)

# Encode a trajectory — do this once and reuse
encoding = localizer.encode_trajectory("path/to/trajectory.mp4")

# Localize a query image
result = localizer.localize("query.jpg", encoding)

print(f"Closest trajectory frame: {result.closest_frame()}")
print(f"Visible frames: {result.visible_indices()}")

# Save and reload the trajectory encoding
import torch
torch.save(encoding.to_dict(), "encoding.pt")
encoding = TrajectoryEncoding.from_dict(torch.load("encoding.pt"))
```

`localize()` accepts single images or lists, and single encodings or lists — see `lotis/localizer.py` for the full batching API.

## Navigation Evaluation

The release includes an evaluation runner for already-generated Habitat evaluation datasets.

### Evaluation conditions

Each episode hands the robot a **reference trajectory** — a sequence of RGB frames through a scene — and a **query** view, and asks it to reach the trajectory's goal. A run sweeps a matrix of conditions, each containing many episodes:

- **`--dataset gibson | hm3d`** — which Habitat scene set the episodes come from.
- **`--start on | off`** — where the robot starts. `on`: the query lies on the trajectory; `off`: it starts away from the trajectory and must navigate onto it first.
- **`--direction forward | backward | any`** — which trajectory point is the goal. `forward`: the last frame; `backward`: the first frame; `any`: a specific intermediate point.
- **`--camera matched | cross`** — whether the camera at deployment matches the reference trajectory's camera. `matched`: the same camera; `cross`: a different camera (FOV / height / aspect ratio), plus a random per-frame orientation perturbation applied to each reference-trajectory frame.

With no filters a run expands to every valid combination (`on + any` is skipped), writing each condition to `eval_results/<dataset>/<case>/<method>/`.

### Setup

Install Habitat-Sim 0.3.3:

```bash
uv pip install pip
HEADLESS=True uv pip install --no-build-isolation \
    "habitat-sim @ git+https://github.com/facebookresearch/habitat-sim.git@v0.3.3"
```

### Datasets and scenes

Two assets are required: the **Habitat scene meshes** (Gibson, HM3D) and the **LoTIS evaluation dataset** (the pre-generated trajectory-query episodes).

**Habitat scenes.** Both datasets need the original Habitat scene meshes, each under its own license — see the official [Habitat datasets guide](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md). Download both into one directory and point the evaluator at it with `--scene-data-dir` (or the `HABITAT_DATA_DIR` env var), which overrides the configs' `data_path` so you never edit YAML.

- **Gibson** — accept the Gibson license, then place the Habitat `.glb` scenes at `<scene-data-dir>/scene_datasets/gibson/` (with `gibson.scene_dataset_config.json`).
- **HM3D (v0.2)** — request access, then download the **train** split with the Habitat downloader:
  ```bash
  python -m habitat_sim.utils.datasets_download --uids hm3d_train_v0.2 --data-path <scene-data-dir>
  ```
  which lands at `<scene-data-dir>/versioned_data/hm3d-0.2/hm3d/train/`.

Expected layout:

```
<scene-data-dir>/
  scene_datasets/gibson/                 # Gibson .glb + gibson.scene_dataset_config.json
  versioned_data/hm3d-0.2/hm3d/train/    # HM3D v0.2 train scenes
```

**LoTIS evaluation dataset.** Both datasets ship as tarballs in a Hugging Face dataset repo (`fnnBsch/lotis-eval`). Download, extract, and point each config's `dataset_path` at the extracted folder:

```bash
hf download fnnBsch/lotis-eval --repo-type dataset --local-dir /path/to/lotis-eval
tar xzf /path/to/lotis-eval/gibson.tar.gz -C /path/to/lotis-eval
tar xzf /path/to/lotis-eval/hm3d.tar.gz   -C /path/to/lotis-eval
# gibson.yaml -> dataset_path: /path/to/lotis-eval/gibson
# hm3d.yaml   -> dataset_path: /path/to/lotis-eval/hm3d
```

> **Note — frames are rendered on the fly.** Because the Habitat scene datasets are under licenses that do not permit redistributing rendered imagery, the evaluation dataset ships **poses only, without RGB**. The reference-trajectory and query frames are generated from your local scenes when you run the evaluation (or the explorer below). Reconstructing the frames this way can introduce small variations from the exact images used in the paper, so expect metrics to match closely rather than bit-for-bit.

### Explore the dataset

Since the frames are not shipped, use the explorer to render an episode — its reference trajectory, query view, and goal frame — into a [Rerun](https://rerun.io) viewer:

```bash
uv run python -m nav_eval.explore \
    --dataset gibson --start on --direction forward --camera cross --index 0
```

Change `--start` / `--direction` / `--camera` to pick the condition and `--index` to page through episodes. Requires the Habitat scenes to be set up (same as running the eval).

Run LoTIS on all Gibson evaluation cases:

```bash
uv run python -m nav_eval.run \
    --dataset gibson \
    --method lotis
```

With no `--start` / `--direction` / `--camera` filters this runs the full condition matrix described above.

Filter the matrix when needed:

```bash
uv run python -m nav_eval.run \
    --dataset gibson \
    --method lotis \
    --start off \
    --direction forward \
    --camera cross \
    --max-episodes 8
```

Use `--max-episodes` for quick smoke tests, then remove it for the full run. The terminal shows a live dashboard; the full per-episode log, `results.json`, and the resolved configs are written under `eval_results/<dataset>/<case>/<method>/`.

### Live visualization

Add `--visualize` to stream the run to a [Rerun](https://rerun.io) viewer — the live camera feed, the goal image, and, for LoTIS, the predicted image-space trajectory points overlaid on the current view:

```bash
uv run python -m nav_eval.run --dataset gibson --method lotis --max-episodes 4 --visualize
```

This spawns a viewer locally. On a remote machine, run `rerun` locally, forward the port (`ssh -R 9876:localhost:9876 <remote-host>`), and pass `--rerun-connect rerun+http://127.0.0.1:9876/proxy`.

### Reproducing the paper results

Run the full evaluation matrix on both datasets:

```bash
uv run python -m nav_eval.run --dataset gibson --method lotis
uv run python -m nav_eval.run --dataset hm3d --method lotis
```

Each command expands to all valid start / direction / camera conditions reported in the paper.

### Evaluating your own method

The evaluator takes care of Habitat setup, episode iteration, and metrics, and only asks your method for velocity commands. See [`nav_eval/WRITING_A_METHOD.md`](nav_eval/WRITING_A_METHOD.md) for the interface, the observation and episode fields, and a runnable example.

Compare two result directories:

```bash
uv run python -m nav_eval.compare \
    --reference /path/to/reference/ours \
    --candidate eval_results/gibson/off_start_forward_cross_camera/lotis
```

## Roadmap

- [x] Inference code + Gradio Demo
- [x] Evaluation runner for pre-generated datasets
- [ ] Training code

## Citation

```bibtex
@misc{busch2026learninglocalizereferencetrajectories,
      title={Learning to Localize Reference Trajectories in Image-Space for Visual Navigation}, 
      author={Finn Lukas Busch and Matti Vahs and Quantao Yang and Jesús Gerardo Ortega Peimbert and Yixi Cai and Jana Tumova and Olov Andersson},
      year={2026},
      eprint={2602.18803},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2602.18803}, 
}
```
