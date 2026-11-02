"""
BYOL (Bootstrap Your Own Latent)
Implementation for retinal image pretraining.

Reference: Grill et al., 2020 - https://arxiv.org/abs/2006.07733
"""

from __future__ import annotations

import copy
import inspect
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# Robust albumentations import with torchvision fallback
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    A = None
    ToTensorV2 = None
    HAS_ALBUMENTATIONS = False

from torchvision import transforms as T


def _create_solarize(p: float = 0.2):
    """Create a Solarize transform compatible with both Albumentations 1.x and 2.x."""
    if not HAS_ALBUMENTATIONS:
        return None
    sig = inspect.signature(A.Solarize.__init__)
    if "threshold_range" in sig.parameters:
        return A.Solarize(threshold_range=(0.5, 0.5), p=p)
    return A.Solarize(threshold=128, p=p)


class BYOLAugmentation:
    """
    Data augmentation pipeline for BYOL.
    Generates two distinct augmented views of an input retinal fundus image.
    Supports Albumentations (preferred) with automatic TorchVision fallback.
    """

    def __init__(self, image_size: int = 224):
        self.image_size = image_size
        self.use_albumentations = HAS_ALBUMENTATIONS

        if self.use_albumentations:
            solarize1 = _create_solarize(p=0.0)
            solarize2 = _create_solarize(p=0.2)

            t1_steps = [
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
                A.GaussianBlur(blur_limit=(3, 7), p=0.1),
            ]
            if solarize1 is not None:
                t1_steps.append(solarize1)
            t1_steps.extend([
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
            self.transform1 = A.Compose(t1_steps)

            t2_steps = [
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
                A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            ]
            if solarize2 is not None:
                t2_steps.append(solarize2)
            t2_steps.extend([
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
            self.transform2 = A.Compose(t2_steps)
        else:
            self.tv_transform1 = T.Compose([
                T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8),
                T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.1),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self.tv_transform2 = T.Compose([
                T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8),
                T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.5),
                T.RandomSolarize(threshold=128, p=0.2),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __call__(self, image: Union[np.ndarray, Image.Image, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate two augmented views from an input image.

        Args:
            image: Input image (H, W, C) numpy array, PIL Image, or (C, H, W) Tensor.

        Returns:
            Tuple of (view1, view2) torch Tensors of shape (3, H, W).
        """
        if isinstance(image, torch.Tensor):
            if image.ndim == 3 and image.shape[0] in (1, 3):
                image = image.detach().cpu().permute(1, 2, 0).numpy()
            elif image.ndim == 2:
                image = image.detach().cpu().numpy()
            else:
                image = image.detach().cpu().numpy()

        if isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"))

        if self.use_albumentations:
            np_img = np.asarray(image, dtype=np.uint8)
            view1 = self.transform1(image=np_img)["image"]
            view2 = self.transform2(image=np_img)["image"]
        else:
            pil_img = Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB")
            view1 = self.tv_transform1(pil_img)
            view2 = self.tv_transform2(pil_img)

        return view1, view2


class MLPHead(nn.Module):
    """MLP projection and prediction heads for BYOL."""

    def __init__(self, in_dim: int, hidden_dim: int = 4096, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim > 2:
            x = torch.flatten(x, 1)
        return self.net(x)


class BYOL(nn.Module):
    """
    BYOL model for self-supervised pretraining.

    Args:
        backbone: Feature encoder network
        feature_dim: Output dimension of backbone encoder
        projection_dim: Dimension of projection head output
        hidden_dim: Hidden dimension in projection and prediction heads
        moving_average_decay: EMA decay rate for target network
    """

    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        projection_dim: int = 256,
        hidden_dim: int = 4096,
        moving_average_decay: float = 0.996,
    ):
        super().__init__()

        # Online network
        self.online_encoder = backbone
        self.online_projector = MLPHead(feature_dim, hidden_dim, projection_dim)
        self.online_predictor = MLPHead(projection_dim, hidden_dim, projection_dim)

        # Target network (exponential moving average of online network)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_projector = copy.deepcopy(self.online_projector)

        # Stop gradients for target network
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False

        self.m = moving_average_decay

    @torch.no_grad()
    def update_target_network(self):
        """Update target network parameters with EMA and sync batch normalization buffers."""
        # Parameters EMA
        for online_params, target_params in zip(
            self.online_encoder.parameters(),
            self.target_encoder.parameters(),
        ):
            target_params.data.mul_(self.m).add_(online_params.data, alpha=1.0 - self.m)

        for online_params, target_params in zip(
            self.online_projector.parameters(),
            self.target_projector.parameters(),
        ):
            target_params.data.mul_(self.m).add_(online_params.data, alpha=1.0 - self.m)

        # Buffers synchronization (e.g. BatchNorm running stats)
        for online_buffer, target_buffer in zip(
            self.online_encoder.buffers(),
            self.target_encoder.buffers(),
        ):
            target_buffer.data.copy_(online_buffer.data)

        for online_buffer, target_buffer in zip(
            self.online_projector.buffers(),
            self.target_projector.buffers(),
        ):
            target_buffer.data.copy_(online_buffer.data)

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for two augmented image views.

        Args:
            x1, x2: Augmented views of shape (B, C, H, W)

        Returns:
            Tuple of (online_pred_1, online_pred_2, target_proj_1, target_proj_2)
        """
        # Online network predictions
        online_repr_1 = self.online_encoder(x1)
        online_proj_1 = self.online_projector(online_repr_1)
        online_pred_1 = self.online_predictor(online_proj_1)

        online_repr_2 = self.online_encoder(x2)
        online_proj_2 = self.online_projector(online_repr_2)
        online_pred_2 = self.online_predictor(online_proj_2)

        # Target network projections (no gradient)
        with torch.no_grad():
            target_repr_1 = self.target_encoder(x1)
            target_proj_1 = self.target_projector(target_repr_1)

            target_repr_2 = self.target_encoder(x2)
            target_proj_2 = self.target_projector(target_repr_2)

        return (
            online_pred_1,
            online_pred_2,
            target_proj_1.detach(),
            target_proj_2.detach(),
        )

    @staticmethod
    def regression_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Mean squared error between L2-normalized vectors (equivalent to 2 - 2 * cosine_similarity).

        Args:
            x: Online predictions (B, D)
            y: Target projections (B, D)

        Returns:
            Scalar tensor loss
        """
        x_norm = F.normalize(x, dim=-1, p=2, eps=1e-8)
        y_norm = F.normalize(y, dim=-1, p=2, eps=1e-8)
        return 2.0 - 2.0 * (x_norm * y_norm).sum(dim=-1).mean()

    def compute_loss(
        self,
        online_pred_1: torch.Tensor,
        online_pred_2: torch.Tensor,
        target_proj_1: torch.Tensor,
        target_proj_2: torch.Tensor,
    ) -> torch.Tensor:
        """Compute symmetric loss across both augmented views."""
        loss1 = self.regression_loss(online_pred_1, target_proj_2)
        loss2 = self.regression_loss(online_pred_2, target_proj_1)
        return (loss1 + loss2) / 2.0


def train_byol_epoch(
    model: BYOL,
    dataloader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """
    Train BYOL for one epoch.

    Args:
        model: BYOL model
        dataloader: DataLoader yielding (view1, view2) or raw images
        optimizer: Optimizer
        device: Target compute device

    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        # Flexible handling of dataloader output formats
        if isinstance(batch, (list, tuple)) and len(batch) >= 2 and isinstance(batch[0], torch.Tensor) and isinstance(batch[1], torch.Tensor):
            x1 = batch[0].to(device)
            x2 = batch[1].to(device)
        elif isinstance(batch, (list, tuple)) and isinstance(batch[0], (list, tuple)) and len(batch[0]) >= 2:
            x1 = batch[0][0].to(device)
            x2 = batch[0][1].to(device)
        else:
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            images = images.to(device)
            half = images.shape[0] // 2
            if half > 0 and images.shape[0] % 2 == 0:
                x1, x2 = images[:half], images[half:]
            else:
                x1, x2 = images, images

        # Forward pass
        online_pred_1, online_pred_2, target_proj_1, target_proj_2 = model(x1, x2)

        # Compute loss
        loss = model.compute_loss(online_pred_1, online_pred_2, target_proj_1, target_proj_2)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update target network
        model.update_target_network()

        total_loss += float(loss.item())
        num_batches += 1

    return total_loss / max(num_batches, 1)


if __name__ == "__main__":
    from torchvision.models import resnet50

    # Create backbone without deprecated arguments
    try:
        backbone = resnet50(weights=None)
    except TypeError:
        backbone = resnet50(pretrained=False)
    backbone.fc = nn.Identity()

    # Create BYOL model
    byol_model = BYOL(backbone, feature_dim=2048, projection_dim=256)

    # Test forward pass
    test_x1 = torch.randn(4, 3, 224, 224)
    test_x2 = torch.randn(4, 3, 224, 224)
    pred_1, pred_2, proj_1, proj_2 = byol_model(test_x1, test_x2)

    print(f"Online pred 1 shape: {pred_1.shape}")
    print(f"Target proj 1 shape: {proj_1.shape}")

    # Test loss
    test_loss = byol_model.compute_loss(pred_1, pred_2, proj_1, proj_2)
    print(f"BYOL Loss: {test_loss.item():.4f}")

    # Test Augmentation
    aug = BYOLAugmentation(image_size=224)
    dummy_img = np.full((224, 224, 3), 128, dtype=np.uint8)
    v1, v2 = aug(dummy_img)
    print(f"Augmented view 1 shape: {v1.shape}, view 2 shape: {v2.shape}")
    print("BYOL module tests passed successfully!")
