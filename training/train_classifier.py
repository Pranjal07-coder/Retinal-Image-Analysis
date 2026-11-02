"""
Training script for retinal disease classification.
Supports CNN and Transformer models with optional self-supervised pretraining.
"""

import argparse
import yaml
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from src.data.datasets import (
    DatasetConfig,
    load_classification_dataset,
    get_classification_transforms
)
from src.models.cnn import create_cnn_model
from src.models.transformers import create_transformer_model
from src.utils.metrics import compute_classification_metrics


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for images, labels in tqdm(dataloader, desc="Training"):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    metrics = compute_classification_metrics(
        np.array(all_labels),
        np.array(all_preds)
    )

    return avg_loss, metrics


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    metrics = compute_classification_metrics(
        np.array(all_labels),
        np.array(all_preds)
    )

    return avg_loss, metrics


def main(args):
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load datasets
    train_transform = get_classification_transforms(
        image_size=config['image_size'],
        is_training=True
    )
    val_transform = get_classification_transforms(
        image_size=config['image_size'],
        is_training=False
    )

    train_dataset = load_classification_dataset(
        DatasetConfig(
            name=config['dataset']['name'],
            split='train',
            subset=config['dataset'].get('subset')
        ),
        transform=train_transform
    )

    val_dataset = load_classification_dataset(
        DatasetConfig(
            name=config['dataset']['name'],
            split='validation',
            subset=config['dataset'].get('subset')
        ),
        transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4
    )

    # Create model
    model_type = args.backbone or config.get('backbone', 'resnet50')

    if model_type in ['vgg16', 'resnet50', 'densenet121']:
        model = create_cnn_model(
            backbone=model_type,
            num_classes=config['num_classes'],
            pretrained=config.get('pretrained', True)
        )
    elif model_type in ['vit_base', 'vit_small', 'swin_tiny', 'efficientformer_l1']:
        model = create_transformer_model(
            backbone=model_type,
            num_classes=config['num_classes'],
            pretrained=config.get('pretrained', True)
        )
    else:
        raise ValueError(f"Unknown backbone: {model_type}")

    model = model.to(device)
    print(f"Model: {model_type}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.get('learning_rate', 1e-4),
        weight_decay=config.get('weight_decay', 1e-4)
    )

    # Training loop
    best_val_acc = 0
    results_dir = Path('results/checkpoints')
    results_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['epochs']}")

        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_metrics = validate(
            model, val_loader, criterion, device
        )

        print(f"Train Loss: {train_loss:.4f}, Acc: {train_metrics['accuracy']:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy']:.4f}")

        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
            }, results_dir / f'{model_type}_best.pth')
            print(f"Saved best model with accuracy: {best_val_acc:.4f}")

    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train retinal disease classifier')
    parser.add_argument('--config', type=str, default='configs/baseline_classification.yaml',
                        help='Path to config file')
    parser.add_argument('--backbone', type=str, default=None,
                        help='Model backbone (overrides config)')
    args = parser.parse_args()

    main(args)
