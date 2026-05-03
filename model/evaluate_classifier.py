"""
Evaluation script for exp10_explainable_flow_lower_dropout experiment
Provides detailed evaluation and visualization of model performance
"""

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['PYTHONWARNINGS'] = 'ignore'

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_curve, 
    auc,
    precision_recall_curve
)
import json

# Import configuration and model
from config import *
from model import ExplainableResNet3D

# Import dataset utilities
import sys
sys.path.append('../video_classification')
from improved_dataset import create_improved_dataloaders


def load_model(model_path, device):
    """Load trained model from checkpoint"""
    model = ExplainableResNet3D(num_classes=NUM_CLASSES, pretrained=False, dropout=DROPOUT)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def evaluate_model(model, dataloader, device):
    """Comprehensive model evaluation"""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    all_frame_importance = []
    
    with torch.no_grad():
        for videos, labels, masks in dataloader:
            videos = videos.to(device)
            labels = labels.to(device)
            
            # Get predictions and attention maps
            outputs, attention = model(videos, return_attention=True)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_frame_importance.append(attention['frame_importance'].cpu().numpy())
    
    return {
        'predictions': np.array(all_preds),
        'labels': np.array(all_labels),
        'probabilities': np.array(all_probs),
        'frame_importance': np.concatenate(all_frame_importance, axis=0)
    }


def plot_roc_curve(labels, probs, save_path):
    """Plot ROC curve"""
    fpr, tpr, thresholds = roc_curve(labels, probs[:, 1])
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return roc_auc


def plot_precision_recall_curve(labels, probs, save_path):
    """Plot Precision-Recall curve"""
    precision, recall, thresholds = precision_recall_curve(labels, probs[:, 1])
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2,
             label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return pr_auc


def plot_confusion_matrix_detailed(cm, save_path):
    """Plot detailed confusion matrix with percentages"""
    plt.figure(figsize=(10, 8))
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create annotations with both counts and percentages
    annot = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'
    
    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues',
                xticklabels=['Intact', 'Detached'],
                yticklabels=['Intact', 'Detached'],
                cbar_kws={'label': 'Count'})
    
    plt.title('Confusion Matrix - Counts and Percentages')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_frame_importance_distribution(frame_importance, save_path):
    """Plot distribution of frame importance scores"""
    # Average importance across all samples
    avg_importance = np.mean(frame_importance, axis=0)
    
    plt.figure(figsize=(12, 6))
    
    # Plot average importance per frame
    plt.subplot(1, 2, 1)
    frames = np.arange(len(avg_importance))
    plt.bar(frames, avg_importance, color='steelblue', alpha=0.7)
    plt.xlabel('Frame Index')
    plt.ylabel('Average Importance Score')
    plt.title('Average Frame Importance Across All Samples')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Plot heatmap of all samples
    plt.subplot(1, 2, 2)
    sns.heatmap(frame_importance[:50], cmap='YlOrRd', cbar_kws={'label': 'Importance'})
    plt.xlabel('Frame Index')
    plt.ylabel('Sample Index')
    plt.title('Frame Importance Heatmap (First 50 Samples)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_classification_report(labels, predictions, save_path):
    """Generate and save detailed classification report"""
    report = classification_report(
        labels, predictions,
        target_names=['Intact', 'Detached'],
        digits=3
    )
    
    with open(save_path, 'w') as f:
        f.write("Classification Report\n")
        f.write("="*80 + "\n\n")
        f.write(report)
        f.write("\n" + "="*80 + "\n")
    
    return report


def evaluate_experiment(model_path=None):
    """Main evaluation function"""
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Evaluating experiment: {EXP_NAME}")
    
    # Default model path
    if model_path is None:
        model_path = os.path.join(SAVE_DIR, "models", f"{EXP_NAME}_best.pth")
    
    print(f"Loading model from: {model_path}")
    
    # Load model
    model = load_model(model_path, device)
    print(f"Model loaded successfully")
    
    # Load test data
    print("Loading test dataset...")
    _, _, test_loader, _ = create_improved_dataloaders(
        DATA_DIR, SPLITS_DIR,
        num_frames=NUM_FRAMES,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        use_augmentation=False  # No augmentation for evaluation
    )
    print(f"Test dataset loaded: {len(test_loader.dataset)} samples")
    
    # Evaluate model
    print("\nEvaluating model...")
    results = evaluate_model(model, test_loader, device)
    
    # Calculate metrics
    cm = confusion_matrix(results['labels'], results['predictions'])
    
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"Accuracy: {100 * np.mean(results['predictions'] == results['labels']):.2f}%")
    print(f"\nConfusion Matrix:")
    print(cm)
    
    # Create evaluation directory
    eval_dir = os.path.join(SAVE_DIR, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    
    # Generate plots
    print("\nGenerating evaluation plots...")
    
    # ROC curve
    roc_auc = plot_roc_curve(
        results['labels'], 
        results['probabilities'],
        os.path.join(eval_dir, f"{EXP_NAME}_roc_curve.png")
    )
    print(f"ROC AUC: {roc_auc:.3f}")
    
    # Precision-Recall curve
    pr_auc = plot_precision_recall_curve(
        results['labels'],
        results['probabilities'],
        os.path.join(eval_dir, f"{EXP_NAME}_pr_curve.png")
    )
    print(f"PR AUC: {pr_auc:.3f}")
    
    # Detailed confusion matrix
    plot_confusion_matrix_detailed(
        cm,
        os.path.join(eval_dir, f"{EXP_NAME}_confusion_matrix_detailed.png")
    )
    
    # Frame importance distribution
    plot_frame_importance_distribution(
        results['frame_importance'],
        os.path.join(eval_dir, f"{EXP_NAME}_frame_importance.png")
    )
    
    # Classification report
    report = generate_classification_report(
        results['labels'],
        results['predictions'],
        os.path.join(eval_dir, f"{EXP_NAME}_classification_report.txt")
    )
    print("\n" + report)
    
    # Save detailed results
    eval_results = {
        'model_path': model_path,
        'num_samples': len(results['labels']),
        'accuracy': float(100 * np.mean(results['predictions'] == results['labels'])),
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'confusion_matrix': cm.tolist(),
        'per_class_accuracy': {
            'intact': float(100 * cm[0, 0] / cm[0].sum()),
            'detached': float(100 * cm[1, 1] / cm[1].sum())
        }
    }
    
    results_path = os.path.join(eval_dir, f"{EXP_NAME}_evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    print(f"\nEvaluation results saved to: {eval_dir}")
    print("="*80)
    
    return eval_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate exp13 model')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to model checkpoint (default: use best model)')
    
    args = parser.parse_args()
    
    results = evaluate_experiment(args.model_path)
