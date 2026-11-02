"""
CNN-based models for retinal image classification.
Supports VGG16, ResNet50, DenseNet121 with pretrained weights.
"""

from enum import Enum
from typing import Optional
import torch
import torch.nn as nn
import torchvision.models as models


class CNNBackbone(Enum):
    """Supported CNN backbones"""
    VGG16 = "vgg16"
    RESNET50 = "resnet50"
    DENSENET121 = "densenet121"


class RetinalCNNClassifier(nn.Module):
    """
    CNN-based classifier for retinal disease detection.

    Args:
        backbone: CNN architecture to use
        num_classes: Number of output classes
        pretrained: Use ImageNet pretrained weights
        dropout_rate: Dropout probability
    """

    def __init__(
        self,
        backbone: CNNBackbone,
        num_classes: int,
        pretrained: bool = True,
        dropout_rate: float = 0.5
    ):
        super().__init__()
        self.backbone_name = backbone.value
        self.num_classes = num_classes

        # Load backbone with modern torchvision weights API
        if backbone == CNNBackbone.VGG16:
            weights = models.VGG16_Weights.DEFAULT if pretrained else None
            self.backbone = models.vgg16(weights=weights)
            num_features = self.backbone.classifier[6].in_features
            # Replace classifier
            self.backbone.classifier[6] = nn.Identity()
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(num_features, num_classes)
            )

        elif backbone == CNNBackbone.RESNET50:
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            num_features = self.backbone.fc.in_features
            # Replace final layer
            self.backbone.fc = nn.Identity()
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(num_features, num_classes)
            )

        elif backbone == CNNBackbone.DENSENET121:
            weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
            self.backbone = models.densenet121(weights=weights)
            num_features = self.backbone.classifier.in_features
            # Replace classifier
            self.backbone.classifier = nn.Identity()
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(num_features, num_classes)
            )

        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        features = self.backbone(x)
        output = self.classifier(features)
        return output

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification head"""
        return self.backbone(x)


def create_cnn_model(
    backbone: str,
    num_classes: int,
    pretrained: bool = True,
    dropout_rate: float = 0.5
) -> RetinalCNNClassifier:
    """
    Factory function to create CNN models.

    Args:
        backbone: Name of backbone ('vgg16', 'resnet50', 'densenet121')
        num_classes: Number of output classes
        pretrained: Use ImageNet pretrained weights
        dropout_rate: Dropout probability

    Returns:
        RetinalCNNClassifier instance

    Example:
        >>> model = create_cnn_model('resnet50', num_classes=5)
        >>> output = model(torch.randn(1, 3, 224, 224))
    """
    backbone_enum = CNNBackbone(backbone.lower())
    return RetinalCNNClassifier(
        backbone=backbone_enum,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate
    )


if __name__ == "__main__":
    # Test models
    for backbone in ['vgg16', 'resnet50', 'densenet121']:
        print(f"\nTesting {backbone}...")
        model = create_cnn_model(backbone, num_classes=5, pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
