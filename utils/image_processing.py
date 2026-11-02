"""
Image preprocessing utilities for retinal fundus images and model inference.
"""

from __future__ import annotations
import numpy as np
import torch
from PIL import Image, ImageOps
from typing import Tuple


def prepare_image(
    image: Image.Image,
    target_size: Tuple[int, int] = (224, 224)
) -> Tuple[Image.Image, torch.Tensor]:
    """
    Standardize, resize, and normalize a retinal fundus image for model inference.

    Args:
        image: Input PIL Image
        target_size: Target (width, height) tuple (default 224x224)

    Returns:
        Tuple of (display_image_rgb, normalized_tensor_1CHW)
    """
    # Auto-orient based on EXIF and convert to 3-channel RGB
    image_rgb = ImageOps.exif_transpose(image).convert("RGB")

    # Resize using bilinear interpolation matching training
    resized = image_rgb.resize(target_size, Image.Resampling.BILINEAR)

    # Convert to float tensor scaled to [0, 1]
    pixels = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(pixels).permute(2, 0, 1)

    # ImageNet standard normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    normalized = ((tensor - mean) / std).unsqueeze(0)

    return image_rgb, normalized
