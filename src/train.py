"""
Training script with MLflow experiment tracking.
Trains the Cats vs Dogs classification model and logs all artifacts.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
import json

from data_preprocessing import create_data_splits, get_data_loaders, CatsDogsDataset, get_val_transforms
from model import get_model, get_model_summary


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)

    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')

    return epoch_loss, epoch_acc, precision, recall, f1, all_preds, all_labels


def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Cat', 'Dog'], yticklabels=['Cat', 'Dog'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path):
    """Plot and save training curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    # Accuracy curves
    ax2.plot(train_accs, label='Train Acc')
    ax2.plot(val_accs, label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def train_model(args):
    """Main training function with MLflow tracking."""

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create data splits
    print("Creating data splits...")
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = create_data_splits(
        args.data_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )

    print(f"Dataset splits - Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")

    # Create data loaders
    train_loader, val_loader = get_data_loaders(
        train_paths, train_labels,
        val_paths, val_labels,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Initialize model
    print(f"Initializing {args.model_name} model...")
    model = get_model(args.model_name, num_classes=2, dropout_rate=args.dropout)
    model = model.to(device)

    model_summary = get_model_summary(model)
    print(f"Model: {model_summary['model_class']}")
    print(f"Total parameters: {model_summary['total_parameters']:,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # MLflow tracking
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name=args.run_name):
        # Log parameters
        mlflow.log_params({
            "model_name": args.model_name,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "num_epochs": args.num_epochs,
            "optimizer": "Adam",
            "train_samples": len(train_paths),
            "val_samples": len(val_paths),
            "test_samples": len(test_paths),
            **model_summary
        })

        # Training loop
        best_val_acc = 0.0
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []

        print(f"\nStarting training for {args.num_epochs} epochs...")

        for epoch in range(args.num_epochs):
            print(f"\nEpoch {epoch + 1}/{args.num_epochs}")

            # Train
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            train_losses.append(train_loss)
            train_accs.append(train_acc)

            # Validate
            val_loss, val_acc, val_precision, val_recall, val_f1, val_preds, val_labels = validate(
                model, val_loader, criterion, device
            )
            val_losses.append(val_loss)
            val_accs.append(val_acc)

            # Update learning rate
            scheduler.step(val_loss)

            # Log metrics
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_precision": val_precision,
                "val_recall": val_recall,
                "val_f1": val_f1,
                "learning_rate": optimizer.param_groups[0]['lr']
            }, step=epoch)

            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                os.makedirs(args.model_dir, exist_ok=True)
                model_path = os.path.join(args.model_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'model_summary': model_summary
                }, model_path)
                print(f"Saved best model with val_acc: {val_acc:.4f}")

        # Plot and log artifacts
        os.makedirs('artifacts', exist_ok=True)

        # Confusion matrix
        cm_path = 'artifacts/confusion_matrix.png'
        plot_confusion_matrix(val_labels, val_preds, cm_path)
        mlflow.log_artifact(cm_path)

        # Training curves
        curves_path = 'artifacts/training_curves.png'
        plot_training_curves(train_losses, val_losses, train_accs, val_accs, curves_path)
        mlflow.log_artifact(curves_path)

        # Log final metrics
        mlflow.log_metrics({
            "best_val_acc": best_val_acc,
            "final_train_loss": train_losses[-1],
            "final_val_loss": val_losses[-1]
        })

        # Log model
        mlflow.pytorch.log_model(model, "model")

        # Save model metadata
        metadata = {
            "model_name": args.model_name,
            "best_val_acc": float(best_val_acc),
            "model_summary": model_summary,
            "class_names": ["cat", "dog"]
        }
        metadata_path = 'artifacts/model_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        mlflow.log_artifact(metadata_path)

        print(f"\n✓ Training completed! Best validation accuracy: {best_val_acc:.4f}")
        print(f"✓ Model saved to: {model_path}")
        print(f"✓ MLflow tracking URI: {mlflow.get_tracking_uri()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Cats vs Dogs classifier')

    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data/raw',
                        help='Path to data directory')
    parser.add_argument('--model_dir', type=str, default='models',
                        help='Directory to save models')

    # Model arguments
    parser.add_argument('--model_name', type=str, default='baseline',
                        choices=['baseline', 'simple'],
                        help='Model architecture to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for regularization')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loading workers')

    # Data split arguments
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Ratio of training data')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Ratio of validation data')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                        help='Ratio of test data')

    # MLflow arguments
    parser.add_argument('--experiment_name', type=str, default='cats_vs_dogs',
                        help='MLflow experiment name')
    parser.add_argument('--run_name', type=str, default=None,
                        help='MLflow run name')

    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    train_model(args)
