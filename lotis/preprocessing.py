"""Image preprocessing utilities."""

import os
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def preprocess_image(
    image: Union[str, Path, Image.Image, np.ndarray],
    size: tuple = (224, 224),
) -> torch.Tensor:
    """
    Preprocess an image for the model.

    Args:
        image: Image as path, PIL Image, or numpy array.
            For numpy arrays: must be RGB format with shape (H, W, 3) and dtype uint8.
            BGR arrays (e.g., from cv2.imread) must be converted first:
            `cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)`
        size: Target size (width, height).

    Returns:
        Preprocessed tensor of shape [3, H, W].
    """
    # Load if path
    if isinstance(image, (str, Path)):
        img = Image.open(image).convert("RGB")
    elif isinstance(image, np.ndarray):
        # Assume RGB uint8 - user must convert BGR to RGB beforehand
        if image.dtype != np.uint8:
            raise ValueError(f"numpy array must be uint8, got {image.dtype}")
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError(f"numpy array must be (H, W, 3), got {image.shape}")
        img = Image.fromarray(image, mode="RGB")
    else:
        img = image.convert("RGB")

    # Resize
    img = img.resize(size, Image.Resampling.BILINEAR)

    # Convert to tensor and normalize
    img_tensor = transforms.ToTensor()(img)
    img_tensor = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)(img_tensor)

    return img_tensor


def load_images_from_path(
    path: Union[str, Path],
    max_images: Optional[int] = None,
) -> List[Image.Image]:
    """
    Load images from a video file or directory.

    Args:
        path: Path to video file or directory of images.
        max_images: Maximum number of images to load. If exceeded, uniformly subsample.

    Returns:
        List of PIL Images.
    """
    path = Path(path)

    if path.is_file() and path.suffix.lower() in [".mp4", ".mov", ".avi", ".mkv"]:
        return _load_from_video(path, max_images)
    elif path.is_dir():
        return _load_from_directory(path, max_images)
    elif path.is_file():
        img = Image.open(path).convert("RGB")
        return [img]
    else:
        raise ValueError(f"Invalid path: {path}")


def _load_from_video(
    video_path: Path,
    max_images: Optional[int] = None,
) -> List[Image.Image]:
    """Load frames from a video file."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

    cap.release()

    if max_images and len(frames) > max_images:
        indices = np.linspace(0, len(frames) - 1, max_images, dtype=int)
        frames = [frames[i] for i in indices]

    return frames


def _load_from_directory(
    dir_path: Path,
    max_images: Optional[int] = None,
) -> List[Image.Image]:
    """Load images from a directory."""
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    # List and sort image files
    files = [
        f for f in dir_path.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]

    # Sort by numeric prefix if possible, otherwise alphabetically
    def sort_key(f: Path):
        stem = f.stem
        try:
            return (0, int("".join(filter(str.isdigit, stem)) or "0"), stem)
        except ValueError:
            return (1, 0, stem)

    files = sorted(files, key=sort_key)

    if max_images and len(files) > max_images:
        indices = np.linspace(0, len(files) - 1, max_images, dtype=int)
        files = [files[i] for i in indices]

    images = []
    for f in files:
        img = Image.open(f).convert("RGB")
        img.load()  # Force load to avoid file handle issues
        images.append(img)

    return images
