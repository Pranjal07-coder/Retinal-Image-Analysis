"""
Transformer-based models for retinal image classification.
Supports Vision Transformer (ViT), Swin Transformer, and EfficientFormer.
Uses timm library for pretrained models.
"""

from enum import Enum
from typing import Optional
import torch
import torch.nn as nn
import timm


class TransformerBackbone(Enum):
    """Supported Transformer backbones"""
    VIT_BASE = "vit_base_patch16_224"
    VIT_SMALL = "vit_small_patch16_224"
    SWIN_TINY = "swin_tiny_patch4_window7_224"
    SWIN_SMALL = "swin_small_patch4_window7_224"
    EFFICIENTFORMER_L1 = "efficientformer_l1"


class RetinalTransformerClassifier(nn.Module):
    """
    Transformer-based classifier for retinal disease detection.

    Args:
        backbone: Transformer architecture to use
        num_classes: Number of output classes
        pretrained: Use ImageNet pretrained weights
        dropout_rate: Dropout probability
    """

    def __init__(
        self,
        backbone: TransformerBackbone,
        num_classes: int,
        pretrained: bool = True,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        self.backbone_name = backbone.value
        self.num_classes = num_classes

        # Load backbone from timm
        self.backbone = timm.create_model(
            backbone.value,
            pretrained=pretrained,
            num_classes=0  # Remove classification head
        )

        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            num_features = self.backbone(dummy_input).shape[1]

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        features = self.backbone(x)
        output = self.classifier(features)
        return output

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification head"""
        return self.backbone(x)

    def get_attention_maps(self, x: torch.Tensor):
        """
        Extract attention maps for visualization.
        Only works for ViT models.
        """
        if 'vit' in self.backbone_name:
            # Hook to capture attention weights
            attention_weights = []

            def hook_fn(module, input, output):
                # output is (B, num_heads, num_patches, num_patches)
                attention_weights.append(output)

            # Register hooks on attention layers
            hooks = []
            for name, module in self.backbone.named_modules():
                if 'attn.softmax' in name or isinstance(module, nn.Softmax):
                    hooks.append(module.register_forward_hook(hook_fn))

            # Forward pass
            _ = self.backbone(x)

            # Remove hooks
            for hook in hooks:
                hook.remove()

            return attention_weights
        else:
            raise NotImplementedError(
                f"Attention map extraction not implemented for {self.backbone_name}"
            )


def create_transformer_model(
    backbone: str,
    num_classes: int,
    pretrained: bool = True,
    dropout_rate: float = 0.1
) -> RetinalTransformerClassifier:
    """
    Factory function to create Transformer models.

    Args:
        backbone: Name of backbone
                 ('vit_base', 'vit_small', 'swin_tiny', 'swin_small', 'efficientformer_l1')
        num_classes: Number of output classes
        pretrained: Use ImageNet pretrained weights
        dropout_rate: Dropout probability

    Returns:
        RetinalTransformerClassifier instance

    Example:
        >>> model = create_transformer_model('vit_base', num_classes=5)
        >>> output = model(torch.randn(1, 3, 224, 224))
    """
    # Map short names to full model names
    backbone_map = {
        'vit_base': TransformerBackbone.VIT_BASE,
        'vit_small': TransformerBackbone.VIT_SMALL,
        'swin_tiny': TransformerBackbone.SWIN_TINY,
        'swin_small': TransformerBackbone.SWIN_SMALL,
        'efficientformer_l1': TransformerBackbone.EFFICIENTFORMER_L1
    }

    backbone_enum = backbone_map.get(backbone.lower())
    if backbone_enum is None:
        raise ValueError(
            f"Unknown backbone: {backbone}. "
            f"Available: {list(backbone_map.keys())}"
        )

    return RetinalTransformerClassifier(
        backbone=backbone_enum,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate
    )


if __name__ == "__main__":
    # Test models
    for backbone in ['vit_base', 'swin_tiny']:
        print(f"\nTesting {backbone}...")
        model = create_transformer_model(backbone, num_classes=5, pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
