"""
Rigorous ML Experiment with Real Retinal Disease Dataset
Using RetinaMNIST - Real retinal fundus images for disease classification
Dataset: 1,080 train / 120 val / 400 test samples
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import time
import argparse
from datetime import datetime
from collections import defaultdict
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, cohen_kappa_score,
    precision_recall_fscore_support, roc_auc_score,
    confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

from medmnist import RetinaMNIST
from PIL import Image
from src.models.cnn import create_cnn_model
from src.visualization.interpretability import compute_grad_cam, save_grad_cam

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Configuration
CONFIG = {
    'dataset_name': 'RetinaMNIST',
    'data_root': './data/medmnist',
    'img_size': 28,  # RetinaMNIST native size
    'resize_to': 224,  # Resize for pretrained models
    'n_classes': 5,
    'epochs': 50,
    'batch_size': 32,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'early_stopping_patience': 15,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'class_names': ['0', '1', '2', '3', '4']  # Retinal disease severity grades
}


class RetinaMNISTDataset(Dataset):
    """Wrapper for RetinaMNIST with resizing"""

    def __init__(self, medmnist_dataset, resize_to=224):
        self.dataset = medmnist_dataset
        self.resize_to = resize_to

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]

        # Convert to PIL Image and guarantee RGB
        from PIL import Image
        if isinstance(img, Image.Image):
            img_pil = img.convert('RGB')
        else:
            img_pil = Image.fromarray(np.array(img)).convert('RGB')

        img_resized = img_pil.resize((self.resize_to, self.resize_to), Image.BILINEAR)

        # Convert to tensor and normalize
        img_tensor = torch.from_numpy(np.array(img_resized)).float().permute(2, 0, 1) / 255.0

        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        target = label[0] if hasattr(label, '__getitem__') else label
        return img_tensor, torch.tensor(int(target), dtype=torch.long)


class EarlyStopping:
    """Early stopping to prevent overfitting"""

    def __init__(self, patience=15, min_delta=1e-4, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, score, epoch):
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
        elif self.mode == 'max':
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.best_epoch = epoch
                self.counter = 0
            else:
                self.counter += 1
        else:  # min
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.best_epoch = epoch
                self.counter = 0
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True

        return self.early_stop


def compute_metrics(y_true, y_pred, y_prob=None, n_classes=5):
    """Compute comprehensive classification metrics"""
    metrics = {}

    # Accuracy
    metrics['accuracy'] = float(accuracy_score(y_true, y_pred))

    # Balanced Accuracy (critical for imbalanced classes)
    metrics['balanced_accuracy'] = float(balanced_accuracy_score(y_true, y_pred))

    # Quadratic Weighted Kappa (ordinal medical severity grading standard)
    try:
        qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
        metrics['quadratic_weighted_kappa'] = float(qwk) if not np.isnan(qwk) else 0.0
    except Exception:
        metrics['quadratic_weighted_kappa'] = 0.0

    # Precision, Recall, F1 (macro and weighted)
    for avg in ['macro', 'weighted']:
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average=avg, zero_division=0
        )
        metrics[f'precision_{avg}'] = float(p)
        metrics[f'recall_{avg}'] = float(r)
        metrics[f'f1_{avg}'] = float(f1)

    # Per-class metrics with explicit class labels 0..n_classes-1
    class_labels = list(range(n_classes))
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=class_labels, average=None, zero_division=0
    )
    for i in range(len(class_labels)):
        metrics[f'precision_class_{i}'] = float(p[i])
        metrics[f'recall_class_{i}'] = float(r[i])
        metrics[f'f1_class_{i}'] = float(f1[i])
        metrics[f'support_class_{i}'] = int(support[i])

    # AUC (if probabilities provided)
    if y_prob is not None:
        try:
            metrics['auc_ovr'] = float(roc_auc_score(
                y_true, y_prob, multi_class='ovr', average='macro'
            ))
        except Exception:
            metrics['auc_ovr'] = 0.0

    return metrics


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    losses = []
    all_preds = []
    all_labels = []
    all_probs = []

    pbar = tqdm(dataloader, desc='Training', leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        # Forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        losses.append(loss.item())
        probs = F.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    metrics = compute_metrics(all_labels, all_preds, np.array(all_probs))
    metrics['loss'] = np.mean(losses)

    return metrics


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """Evaluate model"""
    model.eval()
    losses = []
    all_preds = []
    all_labels = []
    all_probs = []

    for images, labels in tqdm(dataloader, desc='Evaluating', leave=False):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        losses.append(loss.item())
        probs = F.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    metrics = compute_metrics(all_labels, all_preds, np.array(all_probs))
    metrics['loss'] = np.mean(losses)

    return metrics, all_labels, all_preds, np.array(all_probs)


def plot_training_history(history, save_path):
    """Plot comprehensive training curves"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Training History - Real RetinaMNIST Dataset', fontsize=16, fontweight='bold')

    epochs = range(1, len(history['train']['loss']) + 1)

    # Loss
    axes[0, 0].plot(epochs, history['train']['loss'], 'b-', label='Train', linewidth=2)
    axes[0, 0].plot(epochs, history['val']['loss'], 'r-', label='Val', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Cross-Entropy Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Accuracy
    axes[0, 1].plot(epochs, history['train']['accuracy'], 'b-', label='Train', linewidth=2)
    axes[0, 1].plot(epochs, history['val']['accuracy'], 'r-', label='Val', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Classification Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # F1 Score (Macro)
    axes[0, 2].plot(epochs, history['train']['f1_macro'], 'b-', label='Train', linewidth=2)
    axes[0, 2].plot(epochs, history['val']['f1_macro'], 'r-', label='Val', linewidth=2)
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('F1 Score')
    axes[0, 2].set_title('Macro F1 Score')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Precision (Macro)
    axes[1, 0].plot(epochs, history['train']['precision_macro'], 'b-', label='Train', linewidth=2)
    axes[1, 0].plot(epochs, history['val']['precision_macro'], 'r-', label='Val', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].set_title('Macro Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Recall (Macro)
    axes[1, 1].plot(epochs, history['train']['recall_macro'], 'b-', label='Train', linewidth=2)
    axes[1, 1].plot(epochs, history['val']['recall_macro'], 'r-', label='Val', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].set_title('Macro Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # AUC
    if 'auc_ovr' in history['val'] and 'auc_ovr' in history['train'] and len(history['val']['auc_ovr']) > 0:
        axes[1, 2].plot(epochs, history['train']['auc_ovr'], 'b-', label='Train', linewidth=2)
        axes[1, 2].plot(epochs, history['val']['auc_ovr'], 'r-', label='Val', linewidth=2)
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('AUC')
        axes[1, 2].set_title('AUC-ROC (One-vs-Rest)')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
    else:
        axes[1, 2].text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=12)
        axes[1, 2].set_title('AUC-ROC (One-vs-Rest)')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - Real RetinaMNIST Data', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_per_class_metrics(metrics, class_names, save_path):
    """Plot per-class performance metrics"""
    n_classes = len(class_names)

    precision = [metrics.get(f'precision_class_{i}', 0.0) for i in range(n_classes)]
    recall = [metrics.get(f'recall_class_{i}', 0.0) for i in range(n_classes)]
    f1 = [metrics.get(f'f1_class_{i}', 0.0) for i in range(n_classes)]

    x = np.arange(n_classes)
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, precision, width, label='Precision', color='#2ecc71')
    ax.bar(x, recall, width, label='Recall', color='#3498db')
    ax.bar(x + width, f1, width, label='F1 Score', color='#e74c3c')

    ax.set_xlabel('Disease Severity Grade', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Performance Metrics - Real RetinaMNIST', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Grade {name}' for name in class_names])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def train_model(model_name, model, train_loader, val_loader, config, save_dir):
    """Train model with full experiment tracking"""
    print(f"\n{'='*70}")
    print(f"Training {model_name} on Real RetinaMNIST Dataset")
    print(f"{'='*70}")

    device = torch.device(config['device'])
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    early_stopping = EarlyStopping(patience=config['early_stopping_patience'], mode='max')

    history = defaultdict(lambda: defaultdict(list))
    best_val_acc = 0
    best_model_state = None

    start_time = time.time()

    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        print("-" * 70)

        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_metrics, _, _, _ = evaluate(model, val_loader, criterion, device)

        # Record history
        for split, metrics in [('train', train_metrics), ('val', val_metrics)]:
            for key, value in metrics.items():
                history[split][key].append(value)

        # Print metrics
        print(f"Train - Loss: {train_metrics['loss']:.4f}, "
              f"Acc: {train_metrics['accuracy']:.4f}, "
              f"F1: {train_metrics['f1_macro']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
              f"Acc: {val_metrics['accuracy']:.4f}, "
              f"F1: {val_metrics['f1_macro']:.4f}")

        # Learning rate scheduling
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_metrics['accuracy'])
        new_lr = optimizer.param_groups[0]['lr']
        if epoch > 0 and new_lr != old_lr:
            print(f"Learning rate adjusted from {old_lr:.6f} to {new_lr:.6f}")

        # Save best model with deep tensor cloning
        if best_model_state is None or val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"✓ New best model (val_acc: {best_val_acc:.4f})")

        # Early stopping
        if early_stopping(val_metrics['accuracy'], epoch):
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            print(f"Best epoch was {early_stopping.best_epoch+1}")
            break

    training_time = time.time() - start_time

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, history, training_time, early_stopping.best_epoch + 1


def parse_args():
    parser = argparse.ArgumentParser(description="Run RetinaMNIST Classification Experiment")
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help="Directory to save experiment results. Defaults to a new directory under experiments/runs/run_<timestamp> to preserve historical results."
    )
    parser.add_argument('--epochs', type=int, default=CONFIG['epochs'], help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=CONFIG['batch_size'], help="Batch size")
    parser.add_argument('--learning_rate', type=float, default=CONFIG['learning_rate'], help="Learning rate")
    parser.add_argument('--device', type=str, default=CONFIG['device'], help="Compute device ('cpu' or 'cuda')")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--models', nargs='+', default=['ResNet50', 'DenseNet121'], help="Models to train ('ResNet50', 'DenseNet121')")
    return parser.parse_args()


def main():
    args = parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = dict(CONFIG)
    config['epochs'] = args.epochs
    config['batch_size'] = args.batch_size
    config['learning_rate'] = args.learning_rate
    config['device'] = args.device

    print("="*70)
    print("RETINAL DISEASE CLASSIFICATION - REAL RETINAMNIST DATASET")
    print("="*70)
    print(f"\nDataset: RetinaMNIST (Real retinal fundus images)")
    print(f"Task: Ordinal regression (5-class disease severity)")
    print(f"Train: 1,080 | Val: 120 | Test: 400 samples")
    print(f"\nConfiguration:")
    for key, value in config.items():
        if key not in ['class_names', 'dataset_name', 'data_root']:
            print(f"  {key}: {value}")

    # Determine results directory: if not specified, save in experiments/runs/run_<timestamp> to preserve results_real
    if args.output_dir:
        results_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = Path('experiments') / 'runs' / f"run_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving run outputs to: {results_dir.resolve()}")

    # Ensure data root directory exists beforehand to prevent MedMNIST setup RuntimeError
    Path(config['data_root']).mkdir(parents=True, exist_ok=True)

    # Load RetinaMNIST dataset
    print(f"\n[1/8] Loading RetinaMNIST dataset...")
    train_medmnist = RetinaMNIST(split='train', download=True, root=config['data_root'])
    val_medmnist = RetinaMNIST(split='val', download=True, root=config['data_root'])
    test_medmnist = RetinaMNIST(split='test', download=True, root=config['data_root'])

    # Wrap with resize
    train_dataset = RetinaMNISTDataset(train_medmnist, resize_to=config['resize_to'])
    val_dataset = RetinaMNISTDataset(val_medmnist, resize_to=config['resize_to'])
    test_dataset = RetinaMNISTDataset(test_medmnist, resize_to=config['resize_to'])

    print(f"  ✓ Train: {len(train_dataset)} samples")
    print(f"  ✓ Val:   {len(val_dataset)} samples")
    print(f"  ✓ Test:  {len(test_dataset)} samples")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'],
                            shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'],
                             shuffle=False, num_workers=0)

    # Models to test
    available_models = {
        'resnet50': ('ResNet50', lambda: create_cnn_model('resnet50', num_classes=config['n_classes'], pretrained=True)),
        'densenet121': ('DenseNet121', lambda: create_cnn_model('densenet121', num_classes=config['n_classes'], pretrained=True)),
        'vgg16': ('VGG16', lambda: create_cnn_model('vgg16', num_classes=config['n_classes'], pretrained=True)),
    }

    selected_models = []
    for m in args.models:
        m_key = m.lower().strip()
        if m_key in available_models:
            canon_name, builder = available_models[m_key]
            selected_models.append((canon_name, builder()))
        else:
            print(f"Warning: Unknown model '{m}'. Available: {[v[0] for v in available_models.values()]}")

    if not selected_models:
        print("Error: No valid models selected.")
        return

    experiment_results = {}

    for model_name, model in selected_models:
        model_dir = results_dir / model_name.lower()
        model_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[2/8] Training {model_name}...")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Train
        trained_model, history, training_time, best_epoch = train_model(
            model_name, model, train_loader, val_loader, config, model_dir
        )

        # Test evaluation
        print(f"\n[3/8] Evaluating {model_name} on held-out test set...")
        test_metrics, y_true, y_pred, y_prob = evaluate(
            trained_model, test_loader, nn.CrossEntropyLoss(), config['device']
        )

        print(f"\nTest Results:")
        print(f"  Accuracy:          {test_metrics['accuracy']:.4f}")
        print(f"  Balanced Accuracy: {test_metrics.get('balanced_accuracy', 0):.4f}")
        print(f"  QWK:               {test_metrics.get('quadratic_weighted_kappa', 0):.4f}")
        print(f"  Macro F1:          {test_metrics['f1_macro']:.4f}")
        print(f"  AUC-ROC:           {test_metrics.get('auc_ovr', 0):.4f}")

        # Plot training history
        print(f"\n[4/8] Generating training curves...")
        plot_training_history(history, model_dir / 'training_history.png')

        # Plot confusion matrix
        print(f"[5/8] Generating confusion matrix...")
        plot_confusion_matrix(y_true, y_pred, config['class_names'],
                             model_dir / 'confusion_matrix.png')

        # Plot per-class metrics
        print(f"[6/8] Generating per-class metrics...")
        plot_per_class_metrics(test_metrics, config['class_names'],
                              model_dir / 'per_class_metrics.png')

        # Save classification report
        print(f"[7/8] Saving classification report...")
        report = classification_report(y_true, y_pred,
                                      target_names=[f'Grade {c}' for c in config['class_names']],
                                      digits=4)
        with open(model_dir / 'classification_report.txt', 'w') as f:
            f.write(f"RetinaMNIST Test Set Classification Report\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"{'='*70}\n\n")
            f.write(report)

        # Generate Grad-CAM visualizations with auto-detected architecture layer
        print(f"[8/8] Generating Grad-CAM visualizations...")
        gradcam_dir = model_dir / 'gradcam'
        gradcam_dir.mkdir(parents=True, exist_ok=True)

        # Sample one from each class
        for class_idx in range(config['n_classes']):
            for idx, (img, label) in enumerate(test_dataset):
                label_val = label.item() if hasattr(label, 'item') else int(label)
                if label_val == class_idx:
                    try:
                        cam = compute_grad_cam(trained_model, img, device=config['device'])

                        # Denormalize for visualization
                        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                        img_denorm = img * std + mean
                        img_np = img_denorm.cpu().numpy().transpose(1, 2, 0)
                        img_np = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)

                        pred_label = y_pred[idx] if idx < len(y_pred) else 'Unknown'
                        save_grad_cam(
                            img_np, cam,
                            str(gradcam_dir / f'grade_{class_idx}.png'),
                            predicted_class=f'Grade {pred_label}',
                            true_class=f'Grade {class_idx}'
                        )
                    except Exception as e:
                        print(f"Warning: Grad-CAM failed for Grade {class_idx}: {e}")
                    break

        # Save model
        torch.save({
            'model_state_dict': trained_model.state_dict(),
            'test_metrics': test_metrics,
            'history': dict(history),
            'config': config
        }, model_dir / 'model_best.pth')

        # Store results with safe NaN handling for clean JSON serialization
        def _clean_val(v):
            if isinstance(v, (np.floating, float)):
                return float(v) if not np.isnan(v) else 0.0
            if isinstance(v, (np.integer, int)):
                return int(v)
            return v

        experiment_results[model_name] = {
            'test_metrics': {k: _clean_val(v) for k, v in test_metrics.items()},
            'training_time': float(training_time),
            'best_epoch': int(best_epoch),
            'total_epochs': len(history['train']['loss'])
        }

    # Comparison summary
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY - REAL RETINAMNIST DATASET")
    print("="*70)

    for model_name, results in experiment_results.items():
        tm = results['test_metrics']
        print(f"\n{model_name}:")
        print(f"  Best Epoch:        {results['best_epoch']}/{results['total_epochs']}")
        print(f"  Training Time:     {results['training_time']:.1f}s ({results['training_time']/60:.1f}min)")
        print(f"  Test Accuracy:     {tm['accuracy']:.4f}")
        print(f"  Balanced Accuracy: {tm.get('balanced_accuracy', 0):.4f}")
        print(f"  QWK:               {tm.get('quadratic_weighted_kappa', 0):.4f}")
        print(f"  Test F1 (Macro):   {tm['f1_macro']:.4f}")
        print(f"  Test AUC:          {tm.get('auc_ovr', 0):.4f}")

    # Save summary
    with open(results_dir / 'experiment_summary.json', 'w') as f:
        json.dump(experiment_results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Experiment complete! Results saved to: {results_dir}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
