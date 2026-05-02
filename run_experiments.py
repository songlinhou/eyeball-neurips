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

from improved_model import (
    ImprovedResNet3D, 
    MultiScaleResNet3D, 
    ResNet3DWithAuxiliary,
    FocalLoss, 
    LabelSmoothingCrossEntropy
)
from explainable_models import (
    ExplainableResNet3D,
    OpticalFlowResNet3D,
    ExplainableOpticalFlowResNet3D,
    TSMResNet3D
)
from improved_dataset import create_improved_dataloaders, mixup_data, mixup_criterion
import gc

# Experiment configuration
SAVE_DIR = "/content/drive/MyDrive/EyeballProject/classifier_experiment"
DATA_DIR = "../erdes"
SPLITS_DIR = os.path.join(DATA_DIR, "splits", "macula_detached_vs_intact")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create experiment directory structure
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, "models"), exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, "checkpoints"), exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, "logs"), exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, "plots"), exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, "results"), exist_ok=True)

print(f"Experiment results will be saved to: {SAVE_DIR}")
print(f"Using device: {DEVICE}")


class ExperimentLogger:
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
        
        if use_mixup and np.random.rand() < 0.5:
            videos, labels_a, labels_b, lam = mixup_data(videos, labels, alpha=0.2)
            outputs = model(videos)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        else:
            outputs = model(videos)
            loss = criterion(outputs, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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


def run_single_experiment(config, logger):
    """Run a single experiment with given configuration"""
    
    logger.log(f"\n{'='*80}")
    logger.log(f"Configuration: {json.dumps(config, indent=2)}")
    logger.log(f"{'='*80}\n")
    
    # Create dataloaders
    train_loader, val_loader, test_loader, class_weights = create_improved_dataloaders(
        DATA_DIR, SPLITS_DIR, 
        num_frames=config['num_frames'],
        img_size=config['img_size'],
        batch_size=config['batch_size'],
        num_workers=2,
        use_augmentation=config['use_augmentation']
    )
    
    logger.log(f"Dataset loaded: Train={len(train_loader.dataset)}, "
              f"Val={len(val_loader.dataset)}, Test={len(test_loader.dataset)}")
    logger.log(f"Class weights: {class_weights.numpy()}")
    
    # Initialize model
    if config['model_class'] == 'improved':
        model = ImprovedResNet3D(num_classes=2, pretrained=True, 
                                dropout=config['dropout'], use_attention=True)
    elif config['model_class'] == 'multiscale':
        model = MultiScaleResNet3D(num_classes=2, pretrained=True, dropout=config['dropout'])
    elif config['model_class'] == 'auxiliary':
        model = ResNet3DWithAuxiliary(num_classes=2, pretrained=True, dropout=config['dropout'])
    elif config['model_class'] == 'explainable':
        model = ExplainableResNet3D(num_classes=2, pretrained=True, dropout=config['dropout'])
    elif config['model_class'] == 'optical_flow':
        model = OpticalFlowResNet3D(num_classes=2, pretrained=True, dropout=config['dropout'])
    elif config['model_class'] == 'explainable_flow':
        model = ExplainableOpticalFlowResNet3D(num_classes=2, pretrained=True, dropout=config['dropout'])
    elif config['model_class'] == 'tsm':
        model = TSMResNet3D(num_classes=2, pretrained=True, dropout=config['dropout'])
    
    model = model.to(DEVICE)
    logger.log(f"Model: {config['model_class']}, Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss function
    if config['loss_function'] == 'focal':
        criterion = FocalLoss(alpha=class_weights.to(DEVICE), gamma=config['focal_gamma'])
    elif config['loss_function'] == 'label_smoothing':
        criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    
    # Phase 1: Train classifier head only
    logger.log("\n--- PHASE 1: Training classifier head ---")
    if hasattr(model, 'freeze_backbone'):
        model.freeze_backbone()
    
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=config['learning_rate'] * 10, weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    
    phase1_epochs = min(5, config['num_epochs'] // 3)
    best_val_acc = 0.0
    best_model_state = None
    
    for epoch in range(1, phase1_epochs + 1):
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, DEVICE, use_mixup=False)
        val_metrics = validate(model, val_loader, criterion, DEVICE, use_tta=False)
        
        logger.log_epoch(epoch, train_metrics, val_metrics)
        scheduler.step()
        
        if val_metrics['acc'] > best_val_acc:
            best_val_acc = val_metrics['acc']
            best_model_state = copy.deepcopy(model.state_dict())
            # Save best checkpoint in Phase 1
            checkpoint_path = os.path.join(SAVE_DIR, "checkpoints", 
                                          f"{logger.experiment_name}_phase1_best_epoch{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'phase': 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_metrics['acc'],
                'val_loss': val_metrics['loss']
            }, checkpoint_path)
    
    # Phase 2: Fine-tune entire model
    logger.log("\n--- PHASE 2: Fine-tuning entire model ---")
    
    # Clear memory before phase 2
    clear_memory()
    
    if hasattr(model, 'unfreeze_backbone'):
        model.unfreeze_backbone()
    
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], 
                           weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, 
                                                     patience=3)
    
    phase2_epochs = config['num_epochs'] - phase1_epochs
    patience_counter = 0
    patience = 7
    
    for epoch in range(phase1_epochs + 1, config['num_epochs'] + 1):
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, DEVICE, 
                                   use_mixup=config['use_mixup'])
        val_metrics = validate(model, val_loader, criterion, DEVICE, use_tta=False)
        
        logger.log_epoch(epoch, train_metrics, val_metrics)
        scheduler.step(val_metrics['acc'])
        
        # Save checkpoint every 5 epochs
        if epoch % 5 == 0:
            checkpoint_path = os.path.join(SAVE_DIR, "checkpoints", 
                                          f"{logger.experiment_name}_epoch{epoch}.pth")
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
            # Save best model checkpoint in Phase 2
            checkpoint_path = os.path.join(SAVE_DIR, "checkpoints", 
                                          f"{logger.experiment_name}_phase2_best_epoch{epoch}.pth")
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
        
        if patience_counter >= patience:
            logger.log(f"Early stopping triggered at epoch {epoch}")
            break
    
    # Load best model and evaluate on test set
    logger.log("\n--- FINAL EVALUATION ON TEST SET ---")
    model.load_state_dict(best_model_state)
    test_metrics = validate(model, test_loader, criterion, DEVICE, use_tta=config['use_tta'])
    
    logger.log(f"Test Loss: {test_metrics['loss']:.4f}")
    logger.log(f"Test Accuracy: {test_metrics['acc']:.2f}%")
    logger.log(f"Test Precision: {test_metrics['precision']:.3f}")
    logger.log(f"Test Recall: {test_metrics['recall']:.3f}")
    logger.log(f"Test F1: {test_metrics['f1']:.3f}")
    logger.log(f"Test AUC: {test_metrics['auc']:.3f}")
    logger.log(f"Confusion Matrix:\n{test_metrics['confusion_matrix']}")
    
    # Save model
    model_path = os.path.join(SAVE_DIR, "models", f"{logger.experiment_name}_best.pth")
    torch.save(best_model_state, model_path)
    logger.log(f"Model saved to: {model_path}")
    
    # Save history and plots
    logger.save_history()
    logger.plot_history()
    
    # Plot confusion matrix
    plot_confusion_matrix(test_metrics['confusion_matrix'], logger.experiment_name)
    
    return {
        'experiment_name': logger.experiment_name,
        'config': config,
        'best_val_acc': best_val_acc,
        'test_acc': test_metrics['acc'],
        'test_f1': test_metrics['f1'],
        'test_auc': test_metrics['auc'],
        'test_precision': test_metrics['precision'],
        'test_recall': test_metrics['recall']
    }


def plot_confusion_matrix(cm, experiment_name):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Intact', 'Detached'],
                yticklabels=['Intact', 'Detached'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {experiment_name}')
    
    plot_path = os.path.join(SAVE_DIR, "plots", f"{experiment_name}_confusion_matrix.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()


def run_all_experiments():
    """Run comprehensive experiments"""
    
    # Define experiment configurations
    experiments = [
        # {
        #     'name': 'exp01_improved_baseline',
        #     'model_class': 'improved',
        #     'num_frames': 32,
        #     'img_size': 224,
        #     'batch_size': 8,
        #     'num_epochs': 25,
        #     'learning_rate': 1e-4,
        #     'weight_decay': 1e-4,
        #     'dropout': 0.5,
        #     'loss_function': 'focal',
        #     'focal_gamma': 2.0,
        #     'use_augmentation': True,
        #     'use_mixup': True,
        #     'use_tta': True
        # },
        {
            'name': 'exp02_multiscale',
            'model_class': 'multiscale',
            'num_frames': 32,
            'img_size': 224,
            'batch_size': 32,
            'num_epochs': 10,
            'learning_rate': 1e-4,
            'weight_decay': 1e-4,
            'dropout': 0.5,
            'loss_function': 'focal',
            'focal_gamma': 2.0,
            'use_augmentation': True,
            'use_mixup': True,
            'use_tta': True
        },
        {
            'name': 'exp03_auxiliary',
            'model_class': 'auxiliary',
            'num_frames': 32,
            'img_size': 224,
            'batch_size': 32,
            'num_epochs': 10,
            'learning_rate': 1e-4,
            'weight_decay': 1e-4,
            'dropout': 0.5,
            'loss_function': 'focal',
            'focal_gamma': 2.0,
            'use_augmentation': True,
            'use_mixup': True,
            'use_tta': True
        },
        {
            'name': 'exp04_improved_16frames',
            'model_class': 'improved',
            'num_frames': 16,
            'img_size': 224,
            'batch_size': 32,
            'num_epochs': 10,
            'learning_rate': 1e-4,
            'weight_decay': 1e-4,
            'dropout': 0.5,
            'loss_function': 'focal',
            'focal_gamma': 2.0,
            'use_augmentation': True,
            'use_mixup': True,
            'use_tta': True
        },
        {
            'name': 'exp05_improved_64frames',
            'model_class': 'improved',
            'num_frames': 64,
            'img_size': 224,
            'batch_size': 16,
            'num_epochs': 10,
            'learning_rate': 1e-4,
            'weight_decay': 1e-4,
            'dropout': 0.5,
            'loss_function': 'focal',
            'focal_gamma': 2.0,
            'use_augmentation': True,
            'use_mixup': True,
            'use_tta': True
        },
        # {
        #     'name': 'exp06_improved_higher_lr',
        #     'model_class': 'improved',
        #     'num_frames': 32,
        #     'img_size': 224,
        #     'batch_size': 8,
        #     'num_epochs': 25,
        #     'learning_rate': 5e-4,
        #     'weight_decay': 1e-4,
        #     'dropout': 0.5,
        #     'loss_function': 'focal',
        #     'focal_gamma': 2.0,
        #     'use_augmentation': True,
        #     'use_mixup': True,
        #     'use_tta': True
        # },
        # {
        #     'name': 'exp07_improved_lower_dropout',
        #     'model_class': 'improved',
        #     'num_frames': 32,
        #     'img_size': 224,
        #     'batch_size': 8,
        #     'num_epochs': 25,
        #     'learning_rate': 1e-4,
        #     'weight_decay': 1e-4,
        #     'dropout': 0.3,
        #     'loss_function': 'focal',
        #     'focal_gamma': 2.0,
        #     'use_augmentation': True,
        #     'use_mixup': True,
        #     'use_tta': True
        # },
        # New experiments based on exp07
        {
            'name': 'exp08_explainable_lower_dropout',
            'model_class': 'explainable',
            'num_frames': 32,
            'img_size': 224,
            'batch_size': 32,
            'num_epochs': 10,
            'learning_rate': 1e-4,
            'weight_decay': 1e-4,
            'dropout': 0.3,
            'loss_function': 'focal',
            'focal_gamma': 2.0,
            'use_augmentation': True,
            'use_mixup': True,
            'use_tta': True
        },
        {
            'name': 'exp09_optical_flow_lower_dropout',
            'model_class': 'optical_flow',
            'num_frames': 32,
            'img_size': 224,
            'batch_size': 32,
            'num_epochs': 10,
            'learning_rate': 1e-4,
            'weight_decay': 1e-4,
            'dropout': 0.3,
            'loss_function': 'focal',
            'focal_gamma': 2.0,
            'use_augmentation': True,
            'use_mixup': True,
            'use_tta': True
        },
        {
            'name': 'exp10_explainable_flow_lower_dropout',
            'model_class': 'explainable_flow',
            'num_frames': 32,
            'img_size': 224,
            'batch_size': 32,
            'num_epochs': 10,
            'learning_rate': 1e-4,
            'weight_decay': 1e-4,
            'dropout': 0.3,
            'loss_function': 'focal',
            'focal_gamma': 2.0,
            'use_augmentation': True,
            'use_mixup': True,
            'use_tta': True
        },
        {
            'name': 'exp11_tsm_lower_dropout',
            'model_class': 'tsm',
            'num_frames': 32,
            'img_size': 224,
            'batch_size': 32,
            'num_epochs': 10,
            'learning_rate': 1e-4,
            'weight_decay': 1e-4,
            'dropout': 0.3,
            'loss_function': 'focal',
            'focal_gamma': 2.0,
            'use_augmentation': True,
            'use_mixup': True,
            'use_tta': True
        },
        {
            'name': 'exp12_explainable_lower_dropout_16frames',
            'model_class': 'explainable',
            'num_frames': 16,
            'img_size': 224,
            'batch_size': 32,
            'num_epochs': 10,
            'learning_rate': 1e-4,
            'weight_decay': 1e-4,
            'dropout': 0.3,
            'loss_function': 'focal',
            'focal_gamma': 2.0,
            'use_augmentation': True,
            'use_mixup': True,
            'use_tta': True
        },
        {
            'name': 'exp13_explainable_lower_dropout_higher_lr',
            'model_class': 'explainable',
            'num_frames': 32,
            'img_size': 224,
            'batch_size': 32,
            'num_epochs': 10,
            'learning_rate': 2e-4,
            'weight_decay': 1e-4,
            'dropout': 0.3,
            'loss_function': 'focal',
            'focal_gamma': 2.0,
            'use_augmentation': True,
            'use_mixup': True,
            'use_tta': True
        },
        {
            'name': 'exp14_optical_flow_16frames',
            'model_class': 'optical_flow',
            'num_frames': 16,
            'img_size': 224,
            'batch_size': 32,
            'num_epochs': 10,
            'learning_rate': 1e-4,
            'weight_decay': 1e-4,
            'dropout': 0.3,
            'loss_function': 'focal',
            'focal_gamma': 2.0,
            'use_augmentation': True,
            'use_mixup': True,
            'use_tta': True
        },
    ]
    
    all_results = []
    
    print("\n" + "="*80)
    print(f"RUNNING {len(experiments)} EXPERIMENTS")
    print("="*80 + "\n")
    
    for i, config in enumerate(experiments, 1):
        print(f"\n{'='*80}")
        print(f"EXPERIMENT {i}/{len(experiments)}: {config['name']}")
        print(f"{'='*80}\n")
        
        # Clear memory before each experiment
        clear_memory()
        
        start_time = time.time()
        logger = ExperimentLogger(config['name'], SAVE_DIR)
        
        try:
            result = run_single_experiment(config, logger)
            result['training_time_minutes'] = (time.time() - start_time) / 60
            result['status'] = 'completed'
            all_results.append(result)
            
            logger.log(f"\nExperiment completed in {result['training_time_minutes']:.1f} minutes")
            logger.log(f"Best Val Acc: {result['best_val_acc']:.2f}%")
            logger.log(f"Test Acc: {result['test_acc']:.2f}%")
            
        except Exception as e:
            logger.log(f"ERROR: {str(e)}")
            all_results.append({
                'experiment_name': config['name'],
                'config': config,
                'status': 'failed',
                'error': str(e)
            })
        
        # Clear memory after each experiment
        clear_memory()
    
    # Save summary results
    save_experiment_summary(all_results)
    
    return all_results


def save_experiment_summary(results):
    """Save comprehensive summary of all experiments"""
    
    # Create summary DataFrame
    summary_data = []
    for r in results:
        if r['status'] == 'completed':
            summary_data.append({
                'Experiment': r['experiment_name'],
                'Model': r['config']['model_class'],
                'Frames': r['config']['num_frames'],
                'Batch Size': r['config']['batch_size'],
                'Learning Rate': r['config']['learning_rate'],
                'Dropout': r['config']['dropout'],
                'Val Acc (%)': f"{r['best_val_acc']:.2f}",
                'Test Acc (%)': f"{r['test_acc']:.2f}",
                'Test F1': f"{r['test_f1']:.3f}",
                'Test AUC': f"{r['test_auc']:.3f}",
                'Time (min)': f"{r['training_time_minutes']:.1f}"
            })
    
    df_summary = pd.DataFrame(summary_data)
    
    # Save to CSV
    csv_path = os.path.join(SAVE_DIR, "results", "experiment_summary.csv")
    df_summary.to_csv(csv_path, index=False)
    
    # Save detailed results to JSON
    json_path = os.path.join(SAVE_DIR, "results", "experiment_results_detailed.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Create summary report
    report_path = os.path.join(SAVE_DIR, "EXPERIMENT_REPORT.md")
    with open(report_path, 'w') as f:
        f.write("# Ocular Ultrasound Classifier - Experiment Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Total Experiments:** {len(results)}\n")
        f.write(f"**Completed:** {sum(1 for r in results if r['status'] == 'completed')}\n")
        f.write(f"**Failed:** {sum(1 for r in results if r['status'] == 'failed')}\n\n")
        
        f.write("## Summary Table\n\n")
        f.write(df_summary.to_markdown(index=False))
        f.write("\n\n")
        
        # Best model
        if summary_data:
            best_idx = df_summary['Test Acc (%)'].astype(float).idxmax()
            best_exp = summary_data[best_idx]
            f.write("## Best Model\n\n")
            f.write(f"**Experiment:** {best_exp['Experiment']}\n")
            f.write(f"**Model Architecture:** {best_exp['Model']}\n")
            f.write(f"**Test Accuracy:** {best_exp['Test Acc (%)']}%\n")
            f.write(f"**Test F1 Score:** {best_exp['Test F1']}\n")
            f.write(f"**Test AUC:** {best_exp['Test AUC']}\n\n")
        
        f.write("## Experiment Details\n\n")
        for r in results:
            if r['status'] == 'completed':
                f.write(f"### {r['experiment_name']}\n\n")
                f.write(f"- **Model:** {r['config']['model_class']}\n")
                f.write(f"- **Frames:** {r['config']['num_frames']}\n")
                f.write(f"- **Batch Size:** {r['config']['batch_size']}\n")
                f.write(f"- **Learning Rate:** {r['config']['learning_rate']}\n")
                f.write(f"- **Dropout:** {r['config']['dropout']}\n")
                f.write(f"- **Val Accuracy:** {r['best_val_acc']:.2f}%\n")
                f.write(f"- **Test Accuracy:** {r['test_acc']:.2f}%\n")
                f.write(f"- **Test F1:** {r['test_f1']:.3f}\n")
                f.write(f"- **Test AUC:** {r['test_auc']:.3f}\n")
                f.write(f"- **Training Time:** {r['training_time_minutes']:.1f} minutes\n\n")
    
    # Create comparison plots
    create_comparison_plots(df_summary)
    
    print(f"\n{'='*80}")
    print("EXPERIMENT SUMMARY SAVED")
    print(f"{'='*80}")
    print(f"Summary CSV: {csv_path}")
    print(f"Detailed JSON: {json_path}")
    print(f"Report: {report_path}")
    print(f"{'='*80}\n")


def create_comparison_plots(df_summary):
    """Create comparison plots across experiments"""
    
    # Test accuracy comparison
    plt.figure(figsize=(12, 6))
    experiments = df_summary['Experiment'].str.replace('exp\\d+_', '', regex=True)
    test_acc = df_summary['Test Acc (%)'].astype(float)
    
    plt.bar(range(len(experiments)), test_acc, color='steelblue')
    plt.xticks(range(len(experiments)), experiments, rotation=45, ha='right')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Test Accuracy Comparison Across Experiments')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    plot_path = os.path.join(SAVE_DIR, "plots", "accuracy_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Metrics comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = ['Test Acc (%)', 'Test F1', 'Test AUC']
    for idx, metric in enumerate(metrics):
        values = df_summary[metric].astype(float)
        axes[idx].bar(range(len(experiments)), values, color='coral')
        axes[idx].set_xticks(range(len(experiments)))
        axes[idx].set_xticklabels(experiments, rotation=45, ha='right')
        axes[idx].set_ylabel(metric)
        axes[idx].set_title(f'{metric} Comparison')
        axes[idx].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_path = os.path.join(SAVE_DIR, "plots", "metrics_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    print("\n" + "="*80)
    print("OCULAR ULTRASOUND CLASSIFIER - COMPREHENSIVE EXPERIMENTS")
    print("="*80)
    print(f"Save Directory: {SAVE_DIR}")
    print(f"Device: {DEVICE}")
    print("="*80 + "\n")
    
    results = run_all_experiments()
    
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED!")
    print(f"Results saved to: {SAVE_DIR}")
    print("="*80 + "\n")
