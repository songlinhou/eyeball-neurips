import warnings
warnings.filterwarnings('ignore')
import os
os.environ['PYTHONWARNINGS'] = 'ignore'

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../model'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, accuracy_score
import copy
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

from video_models import get_model
from improved_dataset import create_improved_dataloaders
import gc


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VideoClassifierTrainer:
    def __init__(self, model_name, config, save_dir):
        self.model_name = model_name
        self.config = config
        self.save_dir = save_dir
        self.device = DEVICE
        
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "plots"), exist_ok=True)
        
        self.log_file = os.path.join(save_dir, "logs", f"{model_name}_training.log")
        self.history = {
            'train_loss': [], 'train_acc': [], 'train_f1': [],
            'val_loss': [], 'val_acc': [], 'val_f1': []
        }
        
        self._init_log()
        
    def _init_log(self):
        with open(self.log_file, 'w') as f:
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
    
    def log(self, message):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        with open(self.log_file, 'a') as f:
            f.write(log_message + "\n")
    
    def train_epoch(self, model, dataloader, criterion, optimizer):
        model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        for videos, labels, masks in tqdm(dataloader, desc=f"Training {self.model_name}", leave=False):
            videos = videos.to(self.device)
            labels = labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].detach().cpu().numpy())
        
        metrics = {
            'loss': total_loss / len(dataloader),
            'acc': 100. * accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, average='weighted', zero_division=0),
            'recall': recall_score(all_labels, all_preds, average='weighted', zero_division=0),
            'f1': f1_score(all_labels, all_preds, average='weighted', zero_division=0),
        }
        
        try:
            metrics['auc'] = roc_auc_score(all_labels, all_probs)
        except:
            metrics['auc'] = 0.0
        
        return metrics
    
    def validate(self, model, dataloader, criterion):
        model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for videos, labels, masks in tqdm(dataloader, desc=f"Validating {self.model_name}", leave=False):
                videos = videos.to(self.device)
                labels = labels.to(self.device)
                
                outputs = model(videos)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
        
        metrics = {
            'loss': total_loss / len(dataloader),
            'acc': 100. * accuracy_score(all_labels, all_preds),
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
    
    def train(self, train_loader, val_loader, test_loader, class_weights):
        self.log(f"\n{'='*80}")
        self.log(f"Training {self.model_name}")
        self.log(f"Configuration: {json.dumps(self.config, indent=2)}")
        self.log(f"{'='*80}\n")
        
        model = get_model(
            self.model_name,
            num_classes=2,
            pretrained=self.config.get('pretrained', True),
            dropout=self.config.get('dropout', 0.5),
            img_size=self.config.get('img_size', 224),
            num_frames=self.config.get('num_frames', 32)
        )
        model = model.to(self.device)
        
        num_params = sum(p.numel() for p in model.parameters())
        self.log(f"Model parameters: {num_params:,}")
        
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        
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
        
        self.log(f"\n--- PHASE 1: Training classifier head ({phase1_epochs} epochs) ---")
        for epoch in range(1, phase1_epochs + 1):
            train_metrics = self.train_epoch(model, train_loader, criterion, optimizer)
            val_metrics = self.validate(model, val_loader, criterion)
            
            self.log(f"Epoch {epoch}: "
                    f"Train Loss={train_metrics['loss']:.4f}, Acc={train_metrics['acc']:.2f}%, "
                    f"F1={train_metrics['f1']:.3f} | "
                    f"Val Loss={val_metrics['loss']:.4f}, Acc={val_metrics['acc']:.2f}%, "
                    f"F1={val_metrics['f1']:.3f}")
            
            scheduler.step()
            
            if val_metrics['acc'] > best_val_acc:
                best_val_acc = val_metrics['acc']
                best_model_state = copy.deepcopy(model.state_dict())
        
        gc.collect()
        torch.cuda.empty_cache()
        
        self.log(f"\n--- PHASE 2: Fine-tuning entire model ---")
        if hasattr(model, 'unfreeze_backbone'):
            model.unfreeze_backbone()
        
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
        
        phase2_epochs = self.config['num_epochs'] - phase1_epochs
        patience_counter = 0
        patience = 7
        
        for epoch in range(phase1_epochs + 1, self.config['num_epochs'] + 1):
            train_metrics = self.train_epoch(model, train_loader, criterion, optimizer)
            val_metrics = self.validate(model, val_loader, criterion)
            
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['acc'])
            self.history['train_f1'].append(train_metrics['f1'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['acc'])
            self.history['val_f1'].append(val_metrics['f1'])
            
            self.log(f"Epoch {epoch}: "
                    f"Train Loss={train_metrics['loss']:.4f}, Acc={train_metrics['acc']:.2f}%, "
                    f"F1={train_metrics['f1']:.3f}, AUC={train_metrics['auc']:.3f} | "
                    f"Val Loss={val_metrics['loss']:.4f}, Acc={val_metrics['acc']:.2f}%, "
                    f"F1={val_metrics['f1']:.3f}, AUC={val_metrics['auc']:.3f}")
            
            scheduler.step(val_metrics['acc'])
            
            if val_metrics['acc'] > best_val_acc:
                best_val_acc = val_metrics['acc']
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                self.log(f"Early stopping triggered at epoch {epoch}")
                break
        
        self.log("\n--- FINAL EVALUATION ON TEST SET ---")
        model.load_state_dict(best_model_state)
        test_metrics = self.validate(model, test_loader, criterion)
        
        self.log(f"Test Loss: {test_metrics['loss']:.4f}")
        self.log(f"Test Accuracy: {test_metrics['acc']:.2f}%")
        self.log(f"Test Precision: {test_metrics['precision']:.3f}")
        self.log(f"Test Recall: {test_metrics['recall']:.3f}")
        self.log(f"Test F1: {test_metrics['f1']:.3f}")
        self.log(f"Test AUC: {test_metrics['auc']:.3f}")
        self.log(f"Confusion Matrix:\n{test_metrics['confusion_matrix']}")
        
        model_path = os.path.join(self.save_dir, "models", f"{self.model_name}_best.pth")
        torch.save(best_model_state, model_path)
        self.log(f"Model saved to: {model_path}")
        
        self.save_history()
        self.plot_history()
        self.plot_confusion_matrix(test_metrics['confusion_matrix'])
        
        return {
            'model_name': self.model_name,
            'test_acc': test_metrics['acc'],
            'test_precision': test_metrics['precision'],
            'test_recall': test_metrics['recall'],
            'test_f1': test_metrics['f1'],
            'test_auc': test_metrics['auc'],
            'num_params': num_params,
            'confusion_matrix': test_metrics['confusion_matrix'].tolist()
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
        
        axes[0].plot(epochs, self.history['train_loss'], 'b-', label='Training')
        axes[0].plot(epochs, self.history['val_loss'], 'r-', label='Validation')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title(f'{self.model_name} - Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(epochs, self.history['train_acc'], 'b-', label='Training')
        axes[1].plot(epochs, self.history['val_acc'], 'r-', label='Validation')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title(f'{self.model_name} - Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        axes[2].plot(epochs, self.history['train_f1'], 'b-', label='Training')
        axes[2].plot(epochs, self.history['val_f1'], 'r-', label='Validation')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('F1 Score')
        axes[2].set_title(f'{self.model_name} - F1 Score')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(self.save_dir, "plots", f"{self.model_name}_history.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_confusion_matrix(self, cm):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Intact', 'Detached'],
                   yticklabels=['Intact', 'Detached'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix - {self.model_name}')
        
        plot_path = os.path.join(self.save_dir, "plots", f"{self.model_name}_confusion_matrix.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()


def run_benchmark(args):
    DATA_DIR = args.data_dir
    SPLITS_DIR = os.path.join(DATA_DIR, "splits", args.split)
    SAVE_DIR = args.save_dir
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"VIDEO CLASSIFICATION BENCHMARK")
    print(f"{'='*80}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Split: {args.split}")
    print(f"Save directory: {SAVE_DIR}")
    print(f"Device: {DEVICE}")
    print(f"{'='*80}\n")
    
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
    
    print("Loading dataset...")
    train_loader, val_loader, test_loader, class_weights = create_improved_dataloaders(
        DATA_DIR, SPLITS_DIR,
        num_frames=config['num_frames'],
        img_size=config['img_size'],
        batch_size=config['batch_size'],
        num_workers=args.num_workers,
        use_augmentation=True
    )
    
    print(f"Dataset loaded: Train={len(train_loader.dataset)}, "
          f"Val={len(val_loader.dataset)}, Test={len(test_loader.dataset)}")
    print(f"Class weights: {class_weights.numpy()}\n")
    
    models_to_test = args.models if args.models else [
        'resnet3d', 'i3d', 'slowfast', 'x3d', 'mvit', 'videomae', 'timesformer', 'c3d'
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
            trainer = VideoClassifierTrainer(model_name, config, SAVE_DIR)
            result = trainer.train(train_loader, val_loader, test_loader, class_weights)
            result['training_time_minutes'] = (time.time() - start_time) / 60
            result['status'] = 'completed'
            all_results.append(result)
            
            print(f"\n{model_name.upper()} completed in {result['training_time_minutes']:.1f} minutes")
            print(f"Test Accuracy: {result['test_acc']:.2f}%")
            print(f"Test F1: {result['test_f1']:.3f}")
            
        except Exception as e:
            print(f"ERROR training {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            all_results.append({
                'model_name': model_name,
                'status': 'failed',
                'error': str(e)
            })
    
    results_file = os.path.join(SAVE_DIR, "benchmark_results.json")
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {results_file}")
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Video Classification Benchmark')
    parser.add_argument('--data_dir', type=str, default='../../erdes',
                       help='Path to data directory')
    parser.add_argument('--split', type=str, default='macula_detached_vs_intact',
                       help='Dataset split to use')
    parser.add_argument('--save_dir', type=str, default='./results',
                       help='Directory to save results')
    parser.add_argument('--models', nargs='+', default=None,
                       help='Models to test (default: all)')
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
