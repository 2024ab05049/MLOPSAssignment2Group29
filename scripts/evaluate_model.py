"""
Model evaluation script.
Evaluates a trained model on test data and generates detailed metrics.
"""

import os
import argparse
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import json
from tqdm import tqdm

from data_preprocessing import create_data_splits, CatsDogsDataset, get_val_transforms
from model import get_model
from torch.utils.data import DataLoader


def evaluate_model(model, test_loader, device):
    """Evaluate model on test data."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Cat', 'Dog'],
                yticklabels=['Cat', 'Dog'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix to {save_path}")


def plot_roc_curve(y_true, y_probs, save_path):
    """Plot ROC curve."""
    fpr, tpr, thresholds = roc_curve(y_true, y_probs[:, 1])
    roc_auc = roc_auc_score(y_true, y_probs[:, 1])

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved ROC curve to {save_path}")


def plot_prediction_distribution(y_true, y_probs, save_path):
    """Plot prediction confidence distribution."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Confidence distribution for correct predictions
    correct_mask = y_true == np.argmax(y_probs, axis=1)
    correct_conf = np.max(y_probs[correct_mask], axis=1)
    incorrect_conf = np.max(y_probs[~correct_mask], axis=1)

    ax1.hist(correct_conf, bins=30, alpha=0.7, label='Correct', color='green')
    ax1.hist(incorrect_conf, bins=30, alpha=0.7, label='Incorrect', color='red')
    ax1.set_xlabel('Confidence Score')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Prediction Confidence Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Class probability distribution
    cat_probs = y_probs[:, 0]
    dog_probs = y_probs[:, 1]

    ax2.scatter(cat_probs[y_true == 0], dog_probs[y_true == 0],
                alpha=0.5, label='True Cat', color='blue', s=10)
    ax2.scatter(cat_probs[y_true == 1], dog_probs[y_true == 1],
                alpha=0.5, label='True Dog', color='orange', s=10)
    ax2.plot([0, 1], [1, 0], 'k--', alpha=0.3)
    ax2.set_xlabel('Cat Probability')
    ax2.set_ylabel('Dog Probability')
    ax2.set_title('Class Probability Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved prediction distribution to {save_path}")


def main(args):
    """Main evaluation function."""
    print("=" * 60)
    print("Model Evaluation")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print("\nLoading test data...")
    _, _, _, _, test_paths, test_labels = create_data_splits(
        args.data_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )

    transform = get_val_transforms()
    test_dataset = CatsDogsDataset(test_paths, test_labels, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers)

    print(f"Test samples: {len(test_dataset)}")

    # Load model
    print(f"\nLoading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)

    model = get_model("baseline", num_classes=2, dropout_rate=0.5)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print("Model loaded successfully")

    # Evaluate
    print("\nEvaluating model...")
    y_true, y_pred, y_probs = evaluate_model(model, test_loader, device)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')
    roc_auc = roc_auc_score(y_true, y_probs[:, 1])

    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC AUC:   {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred,
                                target_names=['Cat', 'Dog']))

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    results = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc),
        'test_samples': len(test_dataset),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
    }

    results_path = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_path}")

    # Generate plots
    print("\nGenerating visualizations...")
    plot_confusion_matrix(y_true, y_pred,
                         os.path.join(args.output_dir, 'confusion_matrix.png'))
    plot_roc_curve(y_true, y_probs,
                  os.path.join(args.output_dir, 'roc_curve.png'))
    plot_prediction_distribution(y_true, y_probs,
                                os.path.join(args.output_dir, 'prediction_distribution.png'))

    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate trained model')

    parser.add_argument('--model_path', type=str, default='models/best_model.pth',
                        help='Path to trained model')
    parser.add_argument('--data_dir', type=str, default='data/raw',
                        help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default='evaluation',
                        help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loading workers')
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    main(args)
