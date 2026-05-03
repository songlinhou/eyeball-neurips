#!/usr/bin/env python3
"""
Train Multi-Class Explainable ResNet3D Classifier

This script trains the multi-class model on ERDES dataset with balanced train/test splits.
Saves the best model checkpoint based on test accuracy.
"""

import os
import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from tqdm import tqdm

from multiclass_model import create_multiclass_model
from erdes_dataset import ERDESDataset, collate_fn


def get_balanced_splits(dataset, test_size=0.2, random_state=42):
    """
    Create balanced train/test splits
    
    Args:
        dataset: ERDESDataset instance
        test_size: Fraction of data for test set
        random_state: Random seed for reproducibility
        
    Returns:
        train_indices, test_indices
    """
    # Get all labels
    diagnostic_labels = []
    subtype_labels = []
    
    for idx in range(len(dataset)):
        _, labels, _ = dataset[idx]
        diagnostic_labels.append(labels['diagnostic'])
        subtype_labels.append(labels['subtype'])
    
    # Create stratification key (combine diagnostic and subtype)
    stratify_labels = [f"{d}_{s}" for d, s in zip(diagnostic_labels, subtype_labels)]
    
    # Split with stratification
    indices = list(range(len(dataset)))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        stratify=stratify_labels,
        random_state=random_state
    )
    
    return train_indices, test_indices


def train_epoch(model, dataloader, criterion_diagnostic, criterion_subtype, 
                optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0.0
    diagnostic_correct = 0
    subtype_correct = 0
    total_samples = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    for videos, labels, metadata in pbar:
        videos = videos.to(device)
        diagnostic_labels = labels['diagnostic'].to(device)
        subtype_labels = labels['subtype'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(videos)
        
        # Compute losses
        loss_diagnostic = criterion_diagnostic(outputs['diagnostic'], diagnostic_labels)
        loss_subtype = criterion_subtype(outputs['subtype'], subtype_labels)
        
        # Total loss (can weight differently if needed)
        loss = loss_diagnostic + loss_subtype
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item() * videos.size(0)
        
        diagnostic_pred = outputs['diagnostic'].argmax(dim=1)
        subtype_pred = outputs['subtype'].argmax(dim=1)
        
        diagnostic_correct += (diagnostic_pred == diagnostic_labels).sum().item()
        subtype_correct += (subtype_pred == subtype_labels).sum().item()
        total_samples += videos.size(0)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'diag_acc': f'{diagnostic_correct/total_samples:.3f}',
            'subtype_acc': f'{subtype_correct/total_samples:.3f}'
        })
    
    avg_loss = total_loss / total_samples
    diagnostic_acc = diagnostic_correct / total_samples
    subtype_acc = subtype_correct / total_samples
    
    return avg_loss, diagnostic_acc, subtype_acc


def evaluate(model, dataloader, criterion_diagnostic, criterion_subtype, device, split_name='Test'):
    """Evaluate model"""
    model.eval()
    
    total_loss = 0.0
    diagnostic_correct = 0
    subtype_correct = 0
    total_samples = 0
    
    all_diagnostic_preds = []
    all_diagnostic_labels = []
    all_subtype_preds = []
    all_subtype_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f'{split_name}')
        for videos, labels, metadata in pbar:
            videos = videos.to(device)
            diagnostic_labels = labels['diagnostic'].to(device)
            subtype_labels = labels['subtype'].to(device)
            
            # Forward pass
            outputs = model(videos)
            
            # Compute losses
            loss_diagnostic = criterion_diagnostic(outputs['diagnostic'], diagnostic_labels)
            loss_subtype = criterion_subtype(outputs['subtype'], subtype_labels)
            loss = loss_diagnostic + loss_subtype
            
            # Statistics
            total_loss += loss.item() * videos.size(0)
            
            diagnostic_pred = outputs['diagnostic'].argmax(dim=1)
            subtype_pred = outputs['subtype'].argmax(dim=1)
            
            diagnostic_correct += (diagnostic_pred == diagnostic_labels).sum().item()
            subtype_correct += (subtype_pred == subtype_labels).sum().item()
            total_samples += videos.size(0)
            
            # Store predictions
            all_diagnostic_preds.extend(diagnostic_pred.cpu().numpy())
            all_diagnostic_labels.extend(diagnostic_labels.cpu().numpy())
            all_subtype_preds.extend(subtype_pred.cpu().numpy())
            all_subtype_labels.extend(subtype_labels.cpu().numpy())
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'diag_acc': f'{diagnostic_correct/total_samples:.3f}',
                'subtype_acc': f'{subtype_correct/total_samples:.3f}'
            })
    
    avg_loss = total_loss / total_samples
    diagnostic_acc = diagnostic_correct / total_samples
    subtype_acc = subtype_correct / total_samples
    
    return (avg_loss, diagnostic_acc, subtype_acc, 
            all_diagnostic_preds, all_diagnostic_labels,
            all_subtype_preds, all_subtype_labels)


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config = vars(args)
    config['timestamp'] = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n" + "="*60)
    print("Multi-Class Classifier Training")
    print("="*60)
    print(f"CSV Path: {args.csv_path}")
    print(f"Data Root: {args.data_root}")
    print(f"Output Dir: {output_dir}")
    print(f"Test Size: {args.test_size}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning Rate: {args.lr}")
    print("="*60 + "\n")
    
    # Load dataset
    print("Loading dataset...")
    full_dataset = ERDESDataset(
        csv_path=args.csv_path,
        data_root=args.data_root,
        num_frames=args.num_frames,
        img_size=args.img_size
    )
    
    # Create balanced splits
    print("\nCreating balanced train/test splits...")
    train_indices, test_indices = get_balanced_splits(
        full_dataset, 
        test_size=args.test_size,
        random_state=args.random_state
    )
    
    train_dataset = Subset(full_dataset, train_indices)
    test_dataset = Subset(full_dataset, test_indices)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Create model
    print("\nCreating model...")
    model = create_multiclass_model(
        num_diagnostic_classes=args.num_diagnostic_classes,
        num_subtype_classes=args.num_subtype_classes,
        pretrained=args.pretrained,
        dropout=args.dropout
    )
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss functions
    criterion_diagnostic = nn.CrossEntropyLoss()
    criterion_subtype = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    print("\nStarting training...\n")
    best_test_acc = 0.0
    best_epoch = 0
    history = {
        'train_loss': [], 'train_diag_acc': [], 'train_subtype_acc': [],
        'test_loss': [], 'test_diag_acc': [], 'test_subtype_acc': []
    }
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 60)
        
        # Train
        train_loss, train_diag_acc, train_subtype_acc = train_epoch(
            model, train_loader, criterion_diagnostic, criterion_subtype,
            optimizer, device, epoch
        )
        
        # Evaluate
        (test_loss, test_diag_acc, test_subtype_acc,
         diag_preds, diag_labels, subtype_preds, subtype_labels) = evaluate(
            model, test_loader, criterion_diagnostic, criterion_subtype, device
        )
        
        # Overall accuracy (average of both tasks)
        train_acc = (train_diag_acc + train_subtype_acc) / 2
        test_acc = (test_diag_acc + test_subtype_acc) / 2
        
        # Update scheduler
        scheduler.step(test_acc)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_diag_acc'].append(train_diag_acc)
        history['train_subtype_acc'].append(train_subtype_acc)
        history['test_loss'].append(test_loss)
        history['test_diag_acc'].append(test_diag_acc)
        history['test_subtype_acc'].append(test_subtype_acc)
        
        # Print epoch summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train - Loss: {train_loss:.4f}, Diag Acc: {train_diag_acc:.4f}, "
              f"Subtype Acc: {train_subtype_acc:.4f}, Avg Acc: {train_acc:.4f}")
        print(f"  Test  - Loss: {test_loss:.4f}, Diag Acc: {test_diag_acc:.4f}, "
              f"Subtype Acc: {test_subtype_acc:.4f}, Avg Acc: {test_acc:.4f}")
        
        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
                'test_diag_acc': test_diag_acc,
                'test_subtype_acc': test_subtype_acc,
                'config': config
            }
            
            torch.save(checkpoint, output_dir / 'best_model.pth')
            torch.save(model.state_dict(), output_dir / 'best_model_weights.pth')
            
            print(f"  ✓ New best model saved! (Avg Acc: {test_acc:.4f})")
            
            # Save classification reports
            with open(output_dir / 'best_diagnostic_report.txt', 'w') as f:
                f.write("Diagnostic Classification Report\n")
                f.write("="*60 + "\n\n")
                f.write(classification_report(
                    diag_labels, diag_preds,
                    target_names=['non_rd', 'rd']
                ))
            
            with open(output_dir / 'best_subtype_report.txt', 'w') as f:
                f.write("Subtype Classification Report\n")
                f.write("="*60 + "\n\n")
                f.write(classification_report(
                    subtype_labels, subtype_preds,
                    target_names=['normal', 'macula_intact', 'macula_detached', 'pvd']
                ))
        
        # Save latest checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'test_acc': test_acc,
            'config': config
        }
        torch.save(checkpoint, output_dir / 'latest_checkpoint.pth')
    
    # Save training history
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Final summary
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Best Test Accuracy: {best_test_acc:.4f} (Epoch {best_epoch})")
    print(f"Best model saved to: {output_dir / 'best_model_weights.pth'}")
    print("="*60 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Multi-Class Classifier')
    
    # Data arguments
    parser.add_argument('--csv_path', type=str, 
                       default='../benchmarks/input/balanced_split_desc.csv',
                       help='Path to CSV file')
    parser.add_argument('--data_root', type=str,
                       default='../erdes',
                       help='Root directory for video data')
    parser.add_argument('--output_dir', type=str,
                       default='./checkpoints/multiclass',
                       help='Output directory for checkpoints')
    
    # Split arguments
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Fraction of data for test set')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random seed for reproducibility')
    
    # Model arguments
    parser.add_argument('--num_diagnostic_classes', type=int, default=2,
                       help='Number of diagnostic classes')
    parser.add_argument('--num_subtype_classes', type=int, default=4,
                       help='Number of subtype classes')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained weights')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Video processing arguments
    parser.add_argument('--num_frames', type=int, default=32,
                       help='Number of frames to sample')
    parser.add_argument('--img_size', type=int, default=224,
                       help='Image size (height and width)')
    
    args = parser.parse_args()
    
    main(args)
