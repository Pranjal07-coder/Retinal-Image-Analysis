"""
Metrics for classification and segmentation tasks.
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
)
from typing import Dict, Tuple, Optional, Any


def dice_coefficient(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> float:
    """
    Compute Dice coefficient for binary segmentation.

    Args:
        pred: Predicted masks (B, 1, H, W) or (B, H, W)
        target: Ground truth masks (B, 1, H, W) or (B, H, W)
        smooth: Smoothing factor to avoid division by zero

    Returns:
        Dice coefficient
    """
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)

    intersection = (pred * target).sum()
    dice = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)

    return dice.item()


def iou_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> float:
    """
    Compute Intersection over Union (IoU) / Jaccard Index.

    Args:
        pred: Predicted masks
        target: Ground truth masks
        smooth: Smoothing factor

    Returns:
        IoU score
    """
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)

    return iou.item()


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    average: str = 'macro'
) -> Dict[str, Any]:
    """
    Compute classification metrics with medical grading metrics (balanced accuracy, QWK).

    Args:
        y_true: True integer class labels
        y_pred: Predicted integer class labels
        y_prob: Prediction probabilities (for AUC)
        average: Averaging strategy ('micro', 'macro', 'weighted')

    Returns:
        Dictionary of metrics
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    metrics: Dict[str, Any] = {}

    # Standard Accuracy
    metrics['accuracy'] = float(accuracy_score(y_true, y_pred))

    # Balanced Accuracy (critical for imbalanced medical datasets)
    metrics['balanced_accuracy'] = float(balanced_accuracy_score(y_true, y_pred))

    # Quadratic Weighted Kappa (ordinal grading standard for diabetic retinopathy)
    try:
        metrics['quadratic_weighted_kappa'] = float(
            cohen_kappa_score(y_true, y_pred, weights='quadratic')
        )
    except Exception:
        metrics['quadratic_weighted_kappa'] = 0.0

    # Precision, Recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )
    metrics['precision'] = float(precision)
    metrics['recall'] = float(recall)
    metrics['f1'] = float(f1)

    # Also compute macro and weighted specifically for comprehensive reporting
    for avg in ['macro', 'weighted']:
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, average=avg, zero_division=0
        )
        metrics[f'precision_{avg}'] = float(p)
        metrics[f'recall_{avg}'] = float(r)
        metrics[f'f1_{avg}'] = float(f)

    # Per-class metrics
    p_per, r_per, f_per, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    for i in range(len(p_per)):
        metrics[f'precision_class_{i}'] = float(p_per[i])
        metrics[f'recall_class_{i}'] = float(r_per[i])
        metrics[f'f1_class_{i}'] = float(f_per[i])
        metrics[f'support_class_{i}'] = int(support[i])

    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()

    # AUC (if probabilities provided)
    if y_prob is not None:
        try:
            y_prob_arr = np.asarray(y_prob)
            if y_prob_arr.ndim == 1 or y_prob_arr.shape[1] == 1:
                metrics['auc'] = float(roc_auc_score(y_true, y_prob_arr))
            else:
                metrics['auc'] = float(roc_auc_score(
                    y_true, y_prob_arr, multi_class='ovr', average=average
                ))
        except Exception:
            metrics['auc'] = 0.0

    return metrics


def compute_segmentation_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute segmentation metrics.

    Args:
        pred: Predicted masks (B, 1, H, W) with values in [0, 1]
        target: Ground truth masks (B, 1, H, W)
        threshold: Threshold for binary prediction

    Returns:
        Dictionary of metrics
    """
    # Binarize predictions
    pred_binary = (pred > threshold).float()

    metrics = {
        'dice': dice_coefficient(pred_binary, target),
        'iou': iou_score(pred_binary, target)
    }

    return metrics


def compute_metrics(task: str, **kwargs) -> Dict[str, float]:
    """
    Unified interface for computing metrics.

    Args:
        task: 'classification' or 'segmentation'
        **kwargs: Arguments for specific metric functions

    Returns:
        Dictionary of metrics
    """
    if task == 'classification':
        return compute_classification_metrics(**kwargs)
    elif task == 'segmentation':
        return compute_segmentation_metrics(**kwargs)
    else:
        raise ValueError(f"Unknown task: {task}")


if __name__ == "__main__":
    # Test metrics
    print("Testing metrics...")

    # Test Dice coefficient
    pred = torch.rand(2, 1, 64, 64)
    target = torch.rand(2, 1, 64, 64) > 0.5
    dice = dice_coefficient(pred > 0.5, target.float())
    print(f"Dice coefficient: {dice:.4f}")

    # Test IoU
    iou = iou_score(pred > 0.5, target.float())
    print(f"IoU: {iou:.4f}")

    # Test classification metrics
    y_true = np.array([0, 1, 2, 1, 0])
    y_pred = np.array([0, 2, 2, 1, 0])
    y_prob = np.array([[0.8, 0.1, 0.1],
                       [0.2, 0.3, 0.5],
                       [0.1, 0.2, 0.7],
                       [0.2, 0.7, 0.1],
                       [0.9, 0.05, 0.05]])

    metrics = compute_classification_metrics(y_true, y_pred, y_prob)
    print(f"Classification metrics: {metrics}")
