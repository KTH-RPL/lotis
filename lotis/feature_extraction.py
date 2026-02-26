"""Feature extraction using DINOv3 (or DINOv2 fallback)."""
import math
import os
from typing import Optional

import torch
import torch.nn as nn


def setup_feature_extractor(
    model_type: str = "dinov3",
    dinov3_weights: Optional[str] = None,
    dinov3_repo: str = "./dinov3",
) -> nn.Module:
    """
    Set up a feature extractor model.

    Args:
        model_type: "dinov3" (default) or "dinov2".
        dinov3_weights: Path to the DINOv3 weights file (.pth).
            Falls back to the DINOV3_WEIGHTS environment variable.
            Required when model_type="dinov3".
        dinov3_repo: Path to the cloned facebookresearch/dinov3 repository.
            Falls back to the DINOV3_REPO environment variable (default: "./dinov3").

    Returns:
        Feature extractor module (frozen, eval mode).
    """
    if model_type == "dinov2":
        return _setup_dinov2()
    elif model_type == "dinov3":
        weights = dinov3_weights or os.getenv("DINOV3_WEIGHTS")
        repo = os.getenv("DINOV3_REPO", dinov3_repo)
        if not weights:
            raise ValueError(
                "DINOv3 weights path is required. Pass dinov3_weights= to "
                "from_checkpoint() or set the DINOV3_WEIGHTS environment variable."
            )
        return _setup_dinov3(weights, repo)
    else:
        raise ValueError(f"Unknown feature extractor type: {model_type}")


def _setup_dinov3(weights_path: str, repo_path: str) -> nn.Module:
    """Load DINOv3 ViT-B/16 from a local weights file and local repository clone."""
    model = torch.hub.load(
        repo_path,
        "dinov3_vitb16",
        source="local",
        weights=weights_path,
        pretrained=True,
        trust_repo=True,
    )
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model


def _setup_dinov2() -> nn.Module:
    """Load DINOv2 ViT-B/14 from torch hub."""
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model


def extract_features(
    images: torch.Tensor,
    feature_extractor: nn.Module,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Extract patch features from a batch of images.

    Args:
        images: Tensor of shape [B, 3, H, W] with preprocessed images.
        feature_extractor: The feature extractor model.
        device: Device to run extraction on.

    Returns:
        Tensor of shape [B, patch_h, patch_w, feature_dim].
    """
    images = images.to(device)
    batch_size = images.shape[0]

    with torch.no_grad():
        outputs = feature_extractor.forward_features(images)
        patch_features = outputs["x_norm_patchtokens"]  # [B, num_patches, C]
        patch_size = int(math.sqrt(patch_features.shape[1]))
        spatial_features = patch_features.reshape(
            batch_size, patch_size, patch_size, -1
        )

    return spatial_features
