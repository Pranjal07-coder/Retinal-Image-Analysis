"""
Grad-CAM and attention visualization for model interpretability.
Helps understand what regions the model focuses on for disease detection.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List
from pathlib import Path


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).

    Args:
        model: PyTorch model
        target_layer: Layer to compute gradients from (e.g., 'layer4' for ResNet)
    """

    def __init__(self, model: torch.nn.Module, target_layer: str):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks on target layer"""

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        # Find target layer using exact match or dotted suffix to avoid substring false positives
        target_mod = None
        for name, module in self.model.named_modules():
            if name == self.target_layer or name.endswith("." + self.target_layer):
                target_mod = module
                break

        if target_mod is None:
            available = [n for n, m in self.model.named_modules() if isinstance(m, (torch.nn.Conv2d, torch.nn.Sequential, torch.nn.BatchNorm2d))]
            raise ValueError(
                f"Target layer '{self.target_layer}' not found in model. "
                f"Candidate layers include: {available[-5:] if available else 'none'}"
            )

        handle1 = target_mod.register_forward_hook(forward_hook)
        handle2 = target_mod.register_full_backward_hook(backward_hook)
        self.hook_handles.extend([handle1, handle2])

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.

        Args:
            input_tensor: Input image tensor (B, C, H, W)
            target_class: Target class index (if None, uses predicted class)

        Returns:
            CAM heatmap (H, W) with values in [0, 1]
        """
        self.model.eval()

        # Forward pass
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Zero gradients
        self.model.zero_grad()

        # Backward pass for target class
        class_score = output[0, target_class]
        class_score.backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError(
                f"Grad-CAM failed to capture activations/gradients on '{self.target_layer}'."
            )

        # Compute weighted combination of activation maps
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)

        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2), keepdim=True)  # (C, 1, 1)

        # Weighted combination
        cam = (weights * activations).sum(dim=0)  # (H, W)

        # ReLU and normalize
        cam = F.relu(cam)
        cam = cam.cpu().numpy()

        cam_range = cam.max() - cam.min()
        if cam_range > 1e-8:
            cam = (cam - cam.min()) / cam_range
        else:
            cam = np.zeros_like(cam, dtype=np.float32)

        return cam

    def remove_hooks(self):
        """Remove all registered hooks"""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()

    def __del__(self):
        self.remove_hooks()


def get_model_target_layer(model: torch.nn.Module, backbone: Optional[str] = None) -> str:
    """
    Find the appropriate convolutional target layer for Grad-CAM.

    Args:
        model: PyTorch model
        backbone: Architecture identifier (e.g., 'resnet50', 'densenet121', 'vgg16')

    Returns:
        Target layer name string
    """
    backbone_name = (backbone or getattr(model, "backbone_name", "")).lower()
    named_mods = dict(model.named_modules())

    if "resnet" in backbone_name:
        for candidate in ["backbone.layer4", "layer4"]:
            if candidate in named_mods:
                return candidate
    elif "densenet" in backbone_name:
        for candidate in [
            "backbone.features.denseblock4",
            "features.denseblock4",
            "backbone.features.norm5",
            "features.norm5",
        ]:
            if candidate in named_mods:
                return candidate
    elif "vgg" in backbone_name:
        for candidate in ["backbone.features.28", "features.28", "backbone.features", "features"]:
            if candidate in named_mods:
                return candidate

    # General fallback: search for last Conv2d layer
    last_conv = None
    for name, mod in named_mods.items():
        if isinstance(mod, torch.nn.Conv2d):
            last_conv = name
    if last_conv is not None:
        return last_conv

    raise ValueError(f"Could not determine target layer for backbone '{backbone_name}'.")


def compute_grad_cam(
    model: torch.nn.Module,
    image: torch.Tensor,
    target_layer: Optional[str] = None,
    target_class: Optional[int] = None,
    device: str = 'cpu'
) -> np.ndarray:
    """
    Compute Grad-CAM for a single image.

    Args:
        model: PyTorch model
        image: Input image tensor (C, H, W) or (1, C, H, W)
        target_layer: Name of target layer (if None, auto-detected)
        target_class: Target class (None for predicted class)
        device: Device to run on

    Returns:
        CAM heatmap (H, W)
    """
    if target_layer is None:
        target_layer = get_model_target_layer(model)

    model = model.to(device)
    model.eval()

    # Add batch dimension if needed
    if image.ndim == 3:
        image = image.unsqueeze(0)

    image = image.to(device)

    # Create Grad-CAM object
    grad_cam = GradCAM(model, target_layer)

    # Generate CAM
    cam = grad_cam.generate_cam(image, target_class)

    # Cleanup
    grad_cam.remove_hooks()

    return cam


def visualize_grad_cam(
    image: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Overlay Grad-CAM heatmap on original image.

    Args:
        image: Original image (H, W, 3) in RGB, values [0, 255]
        cam: Grad-CAM heatmap (H, W) with values [0, 1]
        alpha: Transparency of heatmap overlay
        colormap: OpenCV colormap

    Returns:
        Overlaid image (H, W, 3)
    """
    # Resize CAM to image size
    h, w = image.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))

    # Convert CAM to heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Ensure image is uint8
    if image.dtype != np.uint8:
        image = np.uint8(image * 255) if image.max() <= 1.0 else np.uint8(image)

    # Overlay heatmap on image
    overlaid = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)

    return overlaid


def save_grad_cam(
    original_image: np.ndarray,
    cam: np.ndarray,
    save_path: str,
    predicted_class: Optional[str] = None,
    true_class: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5)
):
    """
    Save Grad-CAM visualization to file.

    Args:
        original_image: Original image (H, W, 3)
        cam: Grad-CAM heatmap (H, W)
        save_path: Path to save figure
        predicted_class: Predicted class name
        true_class: True class name
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Heatmap only
    axes[1].imshow(cam, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')

    # Overlay
    overlaid = visualize_grad_cam(original_image, cam)
    axes[2].imshow(overlaid)
    title = 'Grad-CAM Overlay'
    if predicted_class:
        title += f'\nPredicted: {predicted_class}'
    if true_class:
        title += f'\nTrue: {true_class}'
    axes[2].set_title(title)
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def batch_grad_cam_visualization(
    model: torch.nn.Module,
    images: List[torch.Tensor],
    target_layer: str,
    save_dir: str,
    class_names: Optional[List[str]] = None,
    device: str = 'cpu'
):
    """
    Generate Grad-CAM visualizations for a batch of images.

    Args:
        model: PyTorch model
        images: List of image tensors
        target_layer: Target layer name
        save_dir: Directory to save visualizations
        class_names: List of class names
        device: Device to run on
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    model = model.to(device)
    model.eval()

    for idx, image in enumerate(images):
        # Compute Grad-CAM
        cam = compute_grad_cam(model, image, target_layer, device=device)

        # Get prediction
        with torch.no_grad():
            if image.ndim == 3:
                image_input = image.unsqueeze(0)
            else:
                image_input = image
            output = model(image_input.to(device))
            pred_class = output.argmax(dim=1).item()

        # Convert image to numpy for visualization
        img_np = image.cpu().numpy()
        if img_np.ndim == 3:
            img_np = np.transpose(img_np, (1, 2, 0))

        # Denormalize if needed (assuming ImageNet normalization)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = img_np * std + mean
        img_np = np.clip(img_np, 0, 1)
        img_np = (img_np * 255).astype(np.uint8)

        # Get class name
        pred_name = class_names[pred_class] if class_names else f"Class {pred_class}"

        # Save visualization
        save_file = save_path / f"gradcam_{idx:03d}.png"
        save_grad_cam(img_np, cam, str(save_file), predicted_class=pred_name)

    print(f"Saved {len(images)} Grad-CAM visualizations to {save_dir}")


if __name__ == "__main__":
    # Test Grad-CAM
    print("Testing Grad-CAM implementation...")

    from torchvision.models import resnet50

    # Create model
    model = resnet50(pretrained=True)
    model.eval()

    # Create random image
    image = torch.randn(1, 3, 224, 224)

    # Compute Grad-CAM
    cam = compute_grad_cam(model, image, 'layer4')

    print(f"CAM shape: {cam.shape}")
    print(f"CAM range: [{cam.min():.3f}, {cam.max():.3f}]")
    print("Grad-CAM test passed!")
