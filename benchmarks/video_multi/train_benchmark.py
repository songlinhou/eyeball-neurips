"""
Multi-class video classification benchmark.

Trains and evaluates models on hierarchical diagnosis classification:
- diagnostic_class: Primary diagnosis (non_rd, rd)
- subtype: Subtype classification (macula_detached, macula_intact, normal, pvd)
"""

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['PYTHONWARNINGS'] = 'ignore'

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    confusion_matrix, accuracy_score, classification_report
)
import copy
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

from multiclass_models import get_multiclass_model
from multiclass_dataset import create_multiclass_dataloaders
import gc


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiClassTrainer:
    """Trainer for multi-class hierarchical video classification"""
    
    def __init__(self, model_name, config, save_dir, label_mappings):
        self.model_name = model_name
        self.config = config
        self.save_dir = save_dir
        self.device = DEVICE
        self.label_mappings = label_mappings
        
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "plots"), exist_ok=True)
        
        self.log_file = os.path.join(save_dir, "logs", f"{model_name}_training.log")
        self.history = {
            'train_loss': [], 'train_diagnostic_acc': [], 'train_subtype_acc': [],
            'val_loss': [], 'val_diagnostic_acc': [], 'val_subtype_acc': []
        }
        
        self._init_log()
    
    def _init_log(self):
        with open(self.log_file, 'w') as f:
            f.write(f"Multi-Class Model: {self.model_name}\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
    
    def log(self, message):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        with open(self.log_file, 'a') as f:
            f.write(log_message + "\n")
    
    def train_epoch(self, model, dataloader, criterion_diagnostic, criterion_subtype, optimizer):
        model.train()
        total_loss = 0.0
        all_diagnostic_preds = []
        all_diagnostic_labels = []
        all_subtype_preds = []
        all_subtype_labels = []
        
        for videos, diagnostic_labels, subtype_labels in tqdm(dataloader, desc=f"Training {self.model_name}", leave=False):
            videos = videos.to(self.device)
            diagnostic_labels = diagnostic_labels.to(self.device)
            subtype_labels = subtype_labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = model(videos)
            
            # Multi-task loss
            loss_diagnostic = criterion_diagnostic(outputs['diagnostic'], diagnostic_labels)
            loss_subtype = criterion_subtype(outputs['subtype'], subtype_labels)
            loss = loss_diagnostic + loss_subtype
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            # Predictions
            _, diagnostic_pred = outputs['diagnostic'].max(1)
            _, subtype_pred = outputs['subtype'].max(1)
            
            all_diagnostic_preds.extend(diagnostic_pred.cpu().numpy())
            all_diagnostic_labels.extend(diagnostic_labels.cpu().numpy())
            all_subtype_preds.extend(subtype_pred.cpu().numpy())
            all_subtype_labels.extend(subtype_labels.cpu().numpy())
        
        metrics = {
            'loss': total_loss / len(dataloader),
            'diagnostic_acc': 100. * accuracy_score(all_diagnostic_labels, all_diagnostic_preds),
            'diagnostic_f1': f1_score(all_diagnostic_labels, all_diagnostic_preds, average='weighted', zero_division=0),
            'subtype_acc': 100. * accuracy_score(all_subtype_labels, all_subtype_preds),
            'subtype_f1': f1_score(all_subtype_labels, all_subtype_preds, average='weighted', zero_division=0)
        }
        
        return metrics
    
    def validate(self, model, dataloader, criterion_diagnostic, criterion_subtype):
        model.eval()
        total_loss = 0.0
        all_diagnostic_preds = []
        all_diagnostic_labels = []
        all_subtype_preds = []
        all_subtype_labels = []
        
        with torch.no_grad():
            for videos, diagnostic_labels, subtype_labels in tqdm(dataloader, desc=f"Validating {self.model_name}", leave=False):
                videos = videos.to(self.device)
                diagnostic_labels = diagnostic_labels.to(self.device)
                subtype_labels = subtype_labels.to(self.device)
                
                outputs = model(videos)
                
                loss_diagnostic = criterion_diagnostic(outputs['diagnostic'], diagnostic_labels)
                loss_subtype = criterion_subtype(outputs['subtype'], subtype_labels)
                loss = loss_diagnostic + loss_subtype
                
                total_loss += loss.item()
                
                _, diagnostic_pred = outputs['diagnostic'].max(1)
                _, subtype_pred = outputs['subtype'].max(1)
                
                all_diagnostic_preds.extend(diagnostic_pred.cpu().numpy())
                all_diagnostic_labels.extend(diagnostic_labels.cpu().numpy())
                all_subtype_preds.extend(subtype_pred.cpu().numpy())
                all_subtype_labels.extend(subtype_labels.cpu().numpy())
        
        metrics = {
            'loss': total_loss / len(dataloader),
            'diagnostic_acc': 100. * accuracy_score(all_diagnostic_labels, all_diagnostic_preds),
            'diagnostic_precision': precision_score(all_diagnostic_labels, all_diagnostic_preds, average='weighted', zero_division=0),
            'diagnostic_recall': recall_score(all_diagnostic_labels, all_diagnostic_preds, average='weighted', zero_division=0),
            'diagnostic_f1': f1_score(all_diagnostic_labels, all_diagnostic_preds, average='weighted', zero_division=0),
            'diagnostic_cm': confusion_matrix(all_diagnostic_labels, all_diagnostic_preds),
            'subtype_acc': 100. * accuracy_score(all_subtype_labels, all_subtype_preds),
            'subtype_precision': precision_score(all_subtype_labels, all_subtype_preds, average='weighted', zero_division=0),
            'subtype_recall': recall_score(all_subtype_labels, all_subtype_preds, average='weighted', zero_division=0),
            'subtype_f1': f1_score(all_subtype_labels, all_subtype_preds, average='weighted', zero_division=0),
            'subtype_cm': confusion_matrix(all_subtype_labels, all_subtype_preds),
            'diagnostic_preds': all_diagnostic_preds,
            'diagnostic_labels': all_diagnostic_labels,
            'subtype_preds': all_subtype_preds,
            'subtype_labels': all_subtype_labels
        }
        
        return metrics
    
    def train(self, train_loader, test_loader, diagnostic_weights, subtype_weights):
        self.log(f"\n{'='*80}")
        self.log(f"Training {self.model_name}")
        self.log(f"Configuration: {json.dumps(self.config, indent=2)}")
        self.log(f"{'='*80}\n")
        
        # Initialize model
        model = get_multiclass_model(
            self.model_name,
            num_diagnostic_classes=self.label_mappings['num_diagnostic_classes'],
            num_subtype_classes=self.label_mappings['num_subtype_classes'],
            pretrained=self.config.get('pretrained', True),
            dropout=self.config.get('dropout', 0.5),
            num_frames=self.config.get('num_frames', 32),
            img_size=self.config.get('img_size', 224)
        )
        model = model.to(self.device)
        
        num_params = sum(p.numel() for p in model.parameters())
        self.log(f"Model parameters: {num_params:,}")
        
        # Loss functions
        criterion_diagnostic = nn.CrossEntropyLoss(weight=diagnostic_weights.to(self.device))
        criterion_subtype = nn.CrossEntropyLoss(weight=subtype_weights.to(self.device))
        
        # Phase 1: Train classifier heads only
        if hasattr(model, 'freeze_backbone'):
            model.freeze_backbone()
        
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.config['learning_rate'] * 10,
            weight_decay=self.config['weight_decay']
        )
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
        
        phase1_epochs = min(5, self.config['num_epochs'] // 3)
        best_val_acc = 0.0
        best_model_state = None
        
        self.log(f"\n--- PHASE 1: Training classifier heads ({phase1_epochs} epochs) ---")
        for epoch in range(1, phase1_epochs + 1):
            train_metrics = self.train_epoch(model, train_loader, criterion_diagnostic, 
                                            criterion_subtype, optimizer)
            val_metrics = self.validate(model, test_loader, criterion_diagnostic, criterion_subtype)
            
            self.log(f"Epoch {epoch}: "
                    f"Train Loss={train_metrics['loss']:.4f}, "
                    f"Diag Acc={train_metrics['diagnostic_acc']:.2f}%, "
                    f"Subtype Acc={train_metrics['subtype_acc']:.2f}% | "
                    f"Val Loss={val_metrics['loss']:.4f}, "
                    f"Diag Acc={val_metrics['diagnostic_acc']:.2f}%, "
                    f"Subtype Acc={val_metrics['subtype_acc']:.2f}%")
            
            scheduler.step()
            
            avg_val_acc = (val_metrics['diagnostic_acc'] + val_metrics['subtype_acc']) / 2
            if avg_val_acc > best_val_acc:
                best_val_acc = avg_val_acc
                best_model_state = copy.deepcopy(model.state_dict())
        
        gc.collect()
        torch.cuda.empty_cache()
        
        # Phase 2: Fine-tune entire model
        self.log(f"\n--- PHASE 2: Fine-tuning entire model ---")
        if hasattr(model, 'unfreeze_backbone'):
            model.unfreeze_backbone()
        
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
        
        patience_counter = 0
        patience = 7
        
        for epoch in range(phase1_epochs + 1, self.config['num_epochs'] + 1):
            train_metrics = self.train_epoch(model, train_loader, criterion_diagnostic,
                                            criterion_subtype, optimizer)
            val_metrics = self.validate(model, test_loader, criterion_diagnostic, criterion_subtype)
            
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_diagnostic_acc'].append(train_metrics['diagnostic_acc'])
            self.history['train_subtype_acc'].append(train_metrics['subtype_acc'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_diagnostic_acc'].append(val_metrics['diagnostic_acc'])
            self.history['val_subtype_acc'].append(val_metrics['subtype_acc'])
            
            self.log(f"Epoch {epoch}: "
                    f"Train Loss={train_metrics['loss']:.4f}, "
                    f"Diag Acc={train_metrics['diagnostic_acc']:.2f}%, "
                    f"Diag F1={train_metrics['diagnostic_f1']:.3f}, "
                    f"Subtype Acc={train_metrics['subtype_acc']:.2f}%, "
                    f"Subtype F1={train_metrics['subtype_f1']:.3f} | "
                    f"Val Loss={val_metrics['loss']:.4f}, "
                    f"Diag Acc={val_metrics['diagnostic_acc']:.2f}%, "
                    f"Diag F1={val_metrics['diagnostic_f1']:.3f}, "
                    f"Subtype Acc={val_metrics['subtype_acc']:.2f}%, "
                    f"Subtype F1={val_metrics['subtype_f1']:.3f}")
            
            avg_val_acc = (val_metrics['diagnostic_acc'] + val_metrics['subtype_acc']) / 2
            scheduler.step(avg_val_acc)
            
            if avg_val_acc > best_val_acc:
                best_val_acc = avg_val_acc
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                self.log(f"Early stopping triggered at epoch {epoch}")
                break
        
        # Final evaluation
        self.log("\n--- FINAL EVALUATION ON TEST SET ---")
        model.load_state_dict(best_model_state)
        test_metrics = self.validate(model, test_loader, criterion_diagnostic, criterion_subtype)
        
        self.log(f"\nDiagnostic Classification:")
        self.log(f"  Accuracy: {test_metrics['diagnostic_acc']:.2f}%")
        self.log(f"  Precision: {test_metrics['diagnostic_precision']:.3f}")
        self.log(f"  Recall: {test_metrics['diagnostic_recall']:.3f}")
        self.log(f"  F1 Score: {test_metrics['diagnostic_f1']:.3f}")
        self.log(f"  Confusion Matrix:\n{test_metrics['diagnostic_cm']}")
        
        self.log(f"\nSubtype Classification:")
        self.log(f"  Accuracy: {test_metrics['subtype_acc']:.2f}%")
        self.log(f"  Precision: {test_metrics['subtype_precision']:.3f}")
        self.log(f"  Recall: {test_metrics['subtype_recall']:.3f}")
        self.log(f"  F1 Score: {test_metrics['subtype_f1']:.3f}")
        self.log(f"  Confusion Matrix:\n{test_metrics['subtype_cm']}")
        
        # Save model
        model_path = os.path.join(self.save_dir, "models", f"{self.model_name}_best.pth")
        torch.save(best_model_state, model_path)
        self.log(f"\nModel saved to: {model_path}")
        
        # Save history and plots
        self.save_history()
        self.plot_history()
        self.plot_confusion_matrices(test_metrics)
        
        return {
            'model_name': self.model_name,
            'num_params': num_params,
            'diagnostic_acc': test_metrics['diagnostic_acc'],
            'diagnostic_precision': test_metrics['diagnostic_precision'],
            'diagnostic_recall': test_metrics['diagnostic_recall'],
            'diagnostic_f1': test_metrics['diagnostic_f1'],
            'diagnostic_cm': test_metrics['diagnostic_cm'].tolist(),
            'subtype_acc': test_metrics['subtype_acc'],
            'subtype_precision': test_metrics['subtype_precision'],
            'subtype_recall': test_metrics['subtype_recall'],
            'subtype_f1': test_metrics['subtype_f1'],
            'subtype_cm': test_metrics['subtype_cm'].tolist()
        }
    
    def save_history(self):
        history_file = os.path.join(self.save_dir, "logs", f"{self.model_name}_history.json")
        with open(history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def plot_history(self):
        if len(self.history['train_loss']) == 0:
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss
        axes[0].plot(epochs, self.history['train_loss'], 'b-', label='Training')
        axes[0].plot(epochs, self.history['val_loss'], 'r-', label='Validation')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title(f'{self.model_name} - Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Diagnostic Accuracy
        axes[1].plot(epochs, self.history['train_diagnostic_acc'], 'b-', label='Training')
        axes[1].plot(epochs, self.history['val_diagnostic_acc'], 'r-', label='Validation')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title(f'{self.model_name} - Diagnostic Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Subtype Accuracy
        axes[2].plot(epochs, self.history['train_subtype_acc'], 'b-', label='Training')
        axes[2].plot(epochs, self.history['val_subtype_acc'], 'r-', label='Validation')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Accuracy (%)')
        axes[2].set_title(f'{self.model_name} - Subtype Accuracy')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(self.save_dir, "plots", f"{self.model_name}_history.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_confusion_matrices(self, metrics):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Diagnostic confusion matrix
        diagnostic_labels = self.label_mappings['diagnostic_classes']
        sns.heatmap(metrics['diagnostic_cm'], annot=True, fmt='d', cmap='Blues',
                   xticklabels=diagnostic_labels, yticklabels=diagnostic_labels,
                   ax=axes[0])
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('True')
        axes[0].set_title(f'{self.model_name} - Diagnostic Classification')
        
        # Subtype confusion matrix
        subtype_labels = self.label_mappings['subtypes']
        sns.heatmap(metrics['subtype_cm'], annot=True, fmt='d', cmap='Blues',
                   xticklabels=subtype_labels, yticklabels=subtype_labels,
                   ax=axes[1])
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('True')
        axes[1].set_title(f'{self.model_name} - Subtype Classification')
        
        plt.tight_layout()
        plot_path = os.path.join(self.save_dir, "plots", f"{self.model_name}_confusion_matrices.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()


def run_benchmark(args):
    DATA_DIR = args.data_dir
    SPLITS_DIR = args.splits_dir
    SAVE_DIR = args.save_dir
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"MULTI-CLASS VIDEO CLASSIFICATION BENCHMARK")
    print(f"{'='*80}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Splits directory: {SPLITS_DIR}")
    print(f"Save directory: {SAVE_DIR}")
    print(f"Device: {DEVICE}")
    print(f"{'='*80}\n")
    
    # Load label mappings
    mappings_file = os.path.join(SPLITS_DIR, 'label_mappings.json')
    with open(mappings_file, 'r') as f:
        label_mappings = json.load(f)
    
    print(f"Label mappings loaded:")
    print(f"  Diagnostic classes: {label_mappings['diagnostic_classes']}")
    print(f"  Subtypes: {label_mappings['subtypes']}")
    
    config = {
        'num_frames': args.num_frames,
        'img_size': args.img_size,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'dropout': args.dropout,
        'pretrained': args.pretrained,
    }
    
    print("\nLoading dataset...")
    train_loader, test_loader, diagnostic_weights, subtype_weights = create_multiclass_dataloaders(
        DATA_DIR, SPLITS_DIR,
        num_frames=config['num_frames'],
        img_size=config['img_size'],
        batch_size=config['batch_size'],
        num_workers=args.num_workers,
        use_augmentation=True
    )
    
    print(f"Dataset loaded: Train={len(train_loader.dataset)}, Test={len(test_loader.dataset)}")
    print(f"Diagnostic class weights: {diagnostic_weights.numpy()}")
    print(f"Subtype class weights: {subtype_weights.numpy()}\n")
    
    models_to_test = args.models if args.models else [
        'resnet3d', 'i3d', 'slowfast', 'x3d', 'mvit', 'videomae', 'timesformer', 'c3d', 'explainable'
    ]
    
    all_results = []
    
    for model_name in models_to_test:
        print(f"\n{'='*80}")
        print(f"Training {model_name.upper()}")
        print(f"{'='*80}\n")
        
        gc.collect()
        torch.cuda.empty_cache()
        
        start_time = time.time()
        
        try:
            trainer = MultiClassTrainer(model_name, config, SAVE_DIR, label_mappings)
            result = trainer.train(train_loader, test_loader, diagnostic_weights, subtype_weights)
            result['training_time_minutes'] = (time.time() - start_time) / 60
            result['status'] = 'completed'
            all_results.append(result)
            
            print(f"\n{model_name.upper()} completed in {result['training_time_minutes']:.1f} minutes")
            print(f"Diagnostic Accuracy: {result['diagnostic_acc']:.2f}%")
            print(f"Subtype Accuracy: {result['subtype_acc']:.2f}%")
            
        except Exception as e:
            print(f"ERROR training {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            all_results.append({
                'model_name': model_name,
                'status': 'failed',
                'error': str(e)
            })
    
    # Save results
    results_file = os.path.join(SAVE_DIR, "benchmark_results.json")
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {results_file}")
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multi-Class Video Classification Benchmark')
    parser.add_argument('--data_dir', type=str, default='../../erdes',
                       help='Path to data directory')
    parser.add_argument('--splits_dir', type=str, default='./splits',
                       help='Directory containing split files')
    parser.add_argument('--save_dir', type=str, default='./results',
                       help='Directory to save results')
    parser.add_argument('--models', nargs='+', default=None,
                       help='Models to test (default: all 9 models - resnet3d, i3d, slowfast, x3d, mvit, videomae, timesformer, c3d, explainable)')
    parser.add_argument('--num_frames', type=int, default=32,
                       help='Number of frames per video')
    parser.add_argument('--img_size', type=int, default=224,
                       help='Image size')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout rate')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained weights')
    
    args = parser.parse_args()
    
    results = run_benchmark(args)
