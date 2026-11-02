"""
SimCLR (Simple Framework for Contrastive Learning of Visual Representations)
Implementation for retinal image pretraining.

Reference: Chen et al., 2020 - https://arxiv.org/abs/2002.05709
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


import inspect


def _create_noise():
    sig = inspect.signature(A.GaussNoise.__init__)
    if "std_range" in sig.parameters:
        return A.GaussNoise(std_range=(0.04, 0.2), p=0.5)
    return A.GaussNoise(var_limit=(10.0, 50.0), p=0.5)


def _create_spatial_transform():
    if hasattr(A, "Affine"):
        return A.Affine(scale=(0.8, 1.2), translate_percent=(-0.1, 0.1), rotate=(-45, 45), p=0.5)
    return A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.5)


class SimCLRAugmentation:
    """
    Data augmentation pipeline for SimCLR.
    Creates two correlated views of each image.
    """

    def __init__(self, image_size: int = 224):
        self.transform = A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            _create_spatial_transform(),
            A.OneOf([
                _create_noise(),
                A.GaussianBlur(blur_limit=(3, 7)),
                A.MotionBlur(blur_limit=5),
            ], p=0.5),
            A.OneOf([
                A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
            ], p=0.8),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

    def __call__(self, image: np.ndarray):
        """Generate two augmented views"""
        view1 = self.transform(image=image)['image']
        view2 = self.transform(image=image)['image']
        return view1, view2


class ProjectionHead(nn.Module):
    """MLP projection head for SimCLR"""

    def __init__(self, in_dim: int, hidden_dim: int = 2048, out_dim: int = 128):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.projection(x)


class SimCLR(nn.Module):
    """
    SimCLR model for self-supervised pretraining.

    Args:
        backbone: Encoder network (e.g., ResNet, ViT)
        projection_dim: Dimension of projection head output
        temperature: Temperature parameter for NT-Xent loss
    """

    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        projection_dim: int = 128,
        temperature: float = 0.5
    ):
        super().__init__()
        self.backbone = backbone
        self.projection_head = ProjectionHead(feature_dim, out_dim=projection_dim)
        self.temperature = temperature

    def forward(self, x1, x2):
        """
        Args:
            x1, x2: Two augmented views of the same batch

        Returns:
            Projected features for both views
        """
        # Extract features
        h1 = self.backbone(x1)
        h2 = self.backbone(x2)

        # Project to contrastive space
        z1 = self.projection_head(h1)
        z2 = self.projection_head(h2)

        return z1, z2

    def nt_xent_loss(self, z1, z2):
        """
        Normalized Temperature-scaled Cross Entropy Loss (NT-Xent).

        Args:
            z1, z2: Projected features from two views (B, projection_dim)

        Returns:
            Contrastive loss
        """
        batch_size = z1.shape[0]

        # Normalize features
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # Concatenate features
        z = torch.cat([z1, z2], dim=0)  # (2B, projection_dim)

        # Compute similarity matrix
        sim_matrix = torch.mm(z, z.t()) / self.temperature  # (2B, 2B)

        # Create labels: positive pairs are (i, i+B) and (i+B, i)
        labels = torch.arange(batch_size, device=z.device)
        labels = torch.cat([labels + batch_size, labels], dim=0)

        # Mask to remove self-similarity
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        sim_matrix = sim_matrix.masked_fill(mask, -9e15)

        # Compute loss
        loss = F.cross_entropy(sim_matrix, labels)

        return loss


def train_simclr_epoch(
    model: SimCLR,
    dataloader,
    optimizer,
    device: torch.device
):
    """
    Train SimCLR for one epoch.

    Args:
        model: SimCLR model
        dataloader: DataLoader with SimCLR augmentation
        optimizer: Optimizer
        device: Device to train on

    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0
    num_batches = 0

    for batch in dataloader:
        images = batch[0].to(device)  # Assuming first element is image

        # Create two views (should be done in dataset/collate_fn)
        # For now, assume augmentation is already applied
        batch_size = images.shape[0] // 2
        x1 = images[:batch_size]
        x2 = images[batch_size:]

        # Forward pass
        z1, z2 = model(x1, x2)

        # Compute loss
        loss = model.nt_xent_loss(z1, z2)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


if __name__ == "__main__":
    # Test SimCLR
    from torchvision.models import resnet50

    # Create backbone
    backbone = resnet50(pretrained=False)
    backbone.fc = nn.Identity()

    # Create SimCLR model
    model = SimCLR(backbone, feature_dim=2048, projection_dim=128)

    # Test forward pass
    x1 = torch.randn(4, 3, 224, 224)
    x2 = torch.randn(4, 3, 224, 224)
    z1, z2 = model(x1, x2)

    print(f"z1 shape: {z1.shape}")
    print(f"z2 shape: {z2.shape}")

    # Test loss
    loss = model.nt_xent_loss(z1, z2)
    print(f"Loss: {loss.item():.4f}")
