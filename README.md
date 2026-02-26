# LoTIS: Learning to Localize Reference Trajectories in Image-Space for Visual Navigation

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2602.18803-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2602.18803)
[![Project Page](https://img.shields.io/badge/Project_Page-blue?style=for-the-badge&logo=googlechrome&logoColor=white)](https://finnbusch.com/lotis)
[![HuggingFace](https://img.shields.io/badge/Demo-HuggingFace-yellow?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/spaces/fnnBsch/lotis-demo)

*Finn Busch, Matti Vahs, Quantao Yang, Jesús Gerardo Ortega Peimbert, Yixi Cai, Jana Tumova, Olov Andersson*

*Division of Robotics, Perception, and Learning at KTH Royal Institute of Technology*

</div>

LoTIS is a model for visual navigation that provides robot-agnostic image-space guidance by localizing a reference RGB trajectory in the robot's current view. Given a reference trajectory (a sequence of RGB images) and a query image from the robot's current viewpoint, LoTIS predicts the 2D image-space coordinates, visibility, and relative distance of each trajectory pose as it would appear in the query view.

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
from lotis import TrajectoryLocalizer

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

## Roadmap

- [x] Inference code + Gradio Demo
- [ ] Evaluation code
- [ ] Training code

## Citation

```bibtex
@article{busch2026lotis,
  title   = {Learning to Localize Reference Trajectories in Image-Space for Visual Navigation},
  author  = {Busch, Finn and Vahs, Matti and Yang, Quantao and {Ortega Peimbert}, {Jes\'{u}s Gerardo} and Cai, Yixi and Tumova, Jana and Andersson, Olov},
  journal = {arXiv preprint arXiv:2602.18803},
  year    = {2026},
}
```
