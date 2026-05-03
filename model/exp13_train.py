"""
Training script for exp10_explainable_flow_lower_dropout experiment
Extracted from video_classification/run_experiments.py
"""

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['PYTHONWARNINGS'] = 'ignore'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import copy
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import gc

# Import configuration and model
from config import *
from model import ExplainableResNet3D

# Import dataset utilities (assuming they exist in parent directory)
import sys
sys.path.append('../video_classification')
from improved_dataset import create_improved_dataloaders, mixup_data, mixup_criterion


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ExperimentLogger:
    """Logger for experiment metrics and history"""
    def __init__(self, experiment_name, save_dir):
        self.experiment_name = experiment_name
        self.save_dir = save_dir
        self.log_file = os.path.join(save_dir, "logs", f"{experiment_name}.log")
        self.metrics_file = os.path.join(save_dir, "logs", f"{experiment_name}_metrics.json")
        self.history = {
            'train_loss': [], 'train_acc': [], 'train_f1': [], 'train_auc': [],
            'val_loss': [], 'val_acc': [], 'val_f1': [], 'val_auc': []
        }
        
        # Initialize log file
        with open(self.log_file, 'w') as f:
            f.write(f"Experiment: {experiment_name}\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
    
    def log(self, message):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        with open(self.log_file, 'a') as f:
            f.write(log_message + "\n")
    
    def log_epoch(self, epoch, train_metrics, val_metrics):
        self.history['train_loss'].append(train_metrics['loss'])
        self.history['train_acc'].append(train_metrics['acc'])
        self.history['train_f1'].append(train_metrics['f1'])
        self.history['train_auc'].append(train_metrics.get('auc', 0))
        
        self.history['val_loss'].append(val_metrics['loss'])
        self.history['val_acc'].append(val_metrics['acc'])
        self.history['val_f1'].append(val_metrics['f1'])
        self.history['val_auc'].append(val_metrics.get('auc', 0))
        
        message = (f"Epoch {epoch}: "
                  f"Train Loss={train_metrics['loss']:.4f}, Acc={train_metrics['acc']:.2f}%, "
                  f"F1={train_metrics['f1']:.3f}, AUC={train_metrics.get('auc', 0):.3f} | "
                  f"Val Loss={val_metrics['loss']:.4f}, Acc={val_metrics['acc']:.2f}%, "
                  f"F1={val_metrics['f1']:.3f}, AUC={val_metrics.get('auc', 0):.3f}")
        self.log(message)
    
    def save_history(self):
        with open(self.metrics_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def plot_history(self):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='Training')
        axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', label='Validation')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[0, 1].plot(epochs, self.history['train_acc'], 'b-', label='Training')
        axes[0, 1].plot(epochs, self.history['val_acc'], 'r-', label='Validation')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # F1 Score
        axes[1, 0].plot(epochs, self.history['train_f1'], 'b-', label='Training')
        axes[1, 0].plot(epochs, self.history['val_f1'], 'r-', label='Validation')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].set_title('Training and Validation F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # AUC
        axes[1, 1].plot(epochs, self.history['train_auc'], 'b-', label='Training')
        axes[1, 1].plot(epochs, self.history['val_auc'], 'r-', label='Validation')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('AUC')
        axes[1, 1].set_title('Training and Validation AUC')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(self.save_dir, "plots", f"{self.experiment_name}_history.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()


def clear_memory():
    """Clear GPU memory"""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def train_epoch(model, dataloader, criterion, optimizer, device, use_mixup=False):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    for batch_idx, (videos, labels, masks) in enumerate(tqdm(dataloader, desc="Training", leave=False)):
        videos = videos.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        if use_mixup and np.random.rand() < MIXUP_PROBABILITY:
            videos, labels_a, labels_b, lam = mixup_data(videos, labels, alpha=MIXUP_ALPHA)
            outputs = model(videos)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        else:
            outputs = model(videos)
            loss = criterion(outputs, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
        optimizer.step()
        
        total_loss += loss.item()
        probs = torch.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs[:, 1].detach().cpu().numpy())
    
    metrics = {
        'loss': total_loss / len(dataloader),
        'acc': 100. * correct / total,
        'precision': precision_score(all_labels, all_preds, average='weighted', zero_division=0),
        'recall': recall_score(all_labels, all_preds, average='weighted', zero_division=0),
        'f1': f1_score(all_labels, all_preds, average='weighted', zero_division=0),
    }
    
    try:
        metrics['auc'] = roc_auc_score(all_labels, all_probs)
    except:
        metrics['auc'] = 0.0
    
    return metrics


def validate(model, dataloader, criterion, device, use_tta=False):
    """Validate model"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for videos, labels, masks in tqdm(dataloader, desc="Validation", leave=False):
            videos = videos.to(device)
            labels = labels.to(device)
            
            if use_tta:
                # Test-time augmentation: average predictions with horizontal flip
                outputs1 = model(videos)
                videos_flip = torch.flip(videos, dims=[4])
                outputs2 = model(videos_flip)
                outputs = (outputs1 + outputs2) / 2
            else:
                outputs = model(videos)
            
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    metrics = {
        'loss': total_loss / len(dataloader),
        'acc': 100. * correct / total,
        'precision': precision_score(all_labels, all_preds, average='weighted', zero_division=0),
        'recall': recall_score(all_labels, all_preds, average='weighted', zero_division=0),
        'f1': f1_score(all_labels, all_preds, average='weighted', zero_division=0),
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }
    
    try:
        metrics['auc'] = roc_auc_score(all_labels, all_probs)
    except:
        metrics['auc'] = 0.0
    
    metrics['confusion_matrix'] = confusion_matrix(all_labels, all_preds)
    
    return metrics


def plot_confusion_matrix(cm, experiment_name, save_dir):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Intact', 'Detached'],
                yticklabels=['Intact', 'Detached'])
    plt.title(f'Confusion Matrix - {experiment_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    plot_path = os.path.join(save_dir, "plots", f"{experiment_name}_confusion_matrix.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()


def train_experiment():
    """Main training function for exp10"""
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Experiment: {EXP_NAME}")
    print(f"Save directory: {SAVE_DIR}")
    
    # Initialize logger
    logger = ExperimentLogger(EXP_NAME, SAVE_DIR)
    logger.log(f"Configuration: {json.dumps(get_config(), indent=2)}")
    
    # Create dataloaders
    logger.log("Loading dataset...")
    train_loader, val_loader, test_loader, class_weights = create_improved_dataloaders(
        DATA_DIR, SPLITS_DIR,
        num_frames=NUM_FRAMES,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        use_augmentation=USE_AUGMENTATION
    )
    
    logger.log(f"Dataset loaded: Train={len(train_loader.dataset)}, "
              f"Val={len(val_loader.dataset)}, Test={len(test_loader.dataset)}")
    logger.log(f"Class weights: {class_weights.numpy()}")
    
    # Initialize model
    logger.log("Initializing model...")
    model = ExplainableResNet3D(num_classes=NUM_CLASSES, pretrained=PRETRAINED, dropout=DROPOUT)
    model = model.to(device)
    logger.log(f"Model: ExplainableOpticalFlowResNet3D, Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss function
    criterion = FocalLoss(alpha=class_weights.to(device), gamma=FOCAL_GAMMA)
    logger.log(f"Loss function: Focal Loss (gamma={FOCAL_GAMMA})")
    
    # PHASE 1: Train classifier head only
    logger.log("\n" + "="*80)
    logger.log("PHASE 1: Training classifier head")
    logger.log("="*80)
    
    model.freeze_backbone()
    
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE * PHASE1_LR_MULTIPLIER,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    
    best_val_acc = 0.0
    best_model_state = None
    
    for epoch in range(1, PHASE1_EPOCHS + 1):
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, use_mixup=False)
        val_metrics = validate(model, val_loader, criterion, device, use_tta=False)
        
        logger.log_epoch(epoch, train_metrics, val_metrics)
        scheduler.step()
        
        if val_metrics['acc'] > best_val_acc:
            best_val_acc = val_metrics['acc']
            best_model_state = copy.deepcopy(model.state_dict())
            
            checkpoint_path = os.path.join(SAVE_DIR, "checkpoints", 
                                          f"{EXP_NAME}_phase1_best_epoch{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'phase': 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_metrics['acc'],
                'val_loss': val_metrics['loss']
            }, checkpoint_path)
    
    # PHASE 2: Fine-tune entire model
    logger.log("\n" + "="*80)
    logger.log("PHASE 2: Fine-tuning entire model")
    logger.log("="*80)
    
    clear_memory()
    model.unfreeze_backbone()
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE
    )
    
    patience_counter = 0
    
    for epoch in range(PHASE1_EPOCHS + 1, NUM_EPOCHS + 1):
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, use_mixup=USE_MIXUP)
        val_metrics = validate(model, val_loader, criterion, device, use_tta=False)
        
        logger.log_epoch(epoch, train_metrics, val_metrics)
        scheduler.step(val_metrics['acc'])
        
        # Save checkpoint every 5 epochs
        if epoch % 5 == 0:
            checkpoint_path = os.path.join(SAVE_DIR, "checkpoints", f"{EXP_NAME}_epoch{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_acc': train_metrics['acc'],
                'val_acc': val_metrics['acc'],
                'train_loss': train_metrics['loss'],
                'val_loss': val_metrics['loss']
            }, checkpoint_path)
            logger.log(f"Checkpoint saved: epoch {epoch}")
        
        if val_metrics['acc'] > best_val_acc:
            best_val_acc = val_metrics['acc']
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            
            checkpoint_path = os.path.join(SAVE_DIR, "checkpoints", 
                                          f"{EXP_NAME}_phase2_best_epoch{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'phase': 2,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_metrics['acc'],
                'val_loss': val_metrics['loss']
            }, checkpoint_path)
        else:
            patience_counter += 1
        
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            logger.log(f"Early stopping triggered at epoch {epoch}")
            break
    
    # Final evaluation on test set
    logger.log("\n" + "="*80)
    logger.log("FINAL EVALUATION ON TEST SET")
    logger.log("="*80)
    
    model.load_state_dict(best_model_state)
    test_metrics = validate(model, test_loader, criterion, device, use_tta=USE_TTA)
    
    logger.log(f"Test Loss: {test_metrics['loss']:.4f}")
    logger.log(f"Test Accuracy: {test_metrics['acc']:.2f}%")
    logger.log(f"Test Precision: {test_metrics['precision']:.3f}")
    logger.log(f"Test Recall: {test_metrics['recall']:.3f}")
    logger.log(f"Test F1: {test_metrics['f1']:.3f}")
    logger.log(f"Test AUC: {test_metrics['auc']:.3f}")
    logger.log(f"Confusion Matrix:\n{test_metrics['confusion_matrix']}")
    
    # Save model
    model_path = os.path.join(SAVE_DIR, "models", f"{EXP_NAME}_best.pth")
    torch.save(best_model_state, model_path)
    logger.log(f"Model saved to: {model_path}")
    
    # Save history and plots
    logger.save_history()
    logger.plot_history()
    plot_confusion_matrix(test_metrics['confusion_matrix'], EXP_NAME, SAVE_DIR)
    
    # Save final results
    results = {
        'experiment_name': EXP_NAME,
        'config': get_config(),
        'best_val_acc': best_val_acc,
        'test_acc': test_metrics['acc'],
        'test_f1': test_metrics['f1'],
        'test_auc': test_metrics['auc'],
        'test_precision': test_metrics['precision'],
        'test_recall': test_metrics['recall'],
        'confusion_matrix': test_metrics['confusion_matrix'].tolist()
    }
    
    results_path = os.path.join(SAVE_DIR, "results", f"{EXP_NAME}_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.log(f"\nExperiment completed successfully!")
    logger.log(f"Results saved to: {results_path}")
    
    return results


if __name__ == "__main__":
    results = train_experiment()
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print(f"Best Validation Accuracy: {results['best_val_acc']:.2f}%")
    print(f"Test Accuracy: {results['test_acc']:.2f}%")
    print(f"Test F1 Score: {results['test_f1']:.3f}")
    print(f"Test AUC: {results['test_auc']:.3f}")
    print("="*80)
