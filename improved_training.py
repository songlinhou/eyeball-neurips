import warnings
warnings.filterwarnings('ignore')
import os
os.environ['PYTHONWARNINGS'] = 'ignore'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import copy

from improved_model import (
    ImprovedResNet3D, 
    MultiScaleResNet3D, 
    ResNet3DWithAuxiliary,
    FocalLoss, 
    LabelSmoothingCrossEntropy
)
from improved_dataset import create_improved_dataloaders, mixup_data, mixup_criterion


DATA_DIR = "../erdes"
SPLITS_DIR = os.path.join(DATA_DIR, "splits", "macula_detached_vs_intact")
NUM_FRAMES = 32
IMG_SIZE = 224
BATCH_SIZE = 8
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self.mode == 'max':
            if score < self.best_score + self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0
        else:
            if score > self.best_score - self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0
        
        return self.early_stop


def train_epoch_basic(model, dataloader, criterion, optimizer, device, use_mixup=False):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    for batch_idx, (videos, labels, masks) in enumerate(tqdm(dataloader, desc="Training")):
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
        
        # Gradient clipping
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
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0
    
    return avg_loss, accuracy, precision, recall, f1, auc


def train_epoch_auxiliary(model, dataloader, criterion, optimizer, device, aux_weight=0.3):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    for batch_idx, (videos, labels, masks) in enumerate(tqdm(dataloader, desc="Training")):
        videos = videos.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        main_outputs, aux_outputs = model(videos, return_aux=True)
        
        main_loss = criterion(main_outputs, labels)
        aux_loss = criterion(aux_outputs, labels)
        loss = main_loss + aux_weight * aux_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = main_outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    return avg_loss, accuracy, precision, recall, f1


def validate(model, dataloader, criterion, device, use_tta=False):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for videos, labels, masks in tqdm(dataloader, desc="Validation"):
            videos = videos.to(device)
            labels = labels.to(device)
            
            if use_tta:
                # Test-time augmentation: horizontal flip
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
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0
    
    cm = confusion_matrix(all_labels, all_preds)
    
    return avg_loss, accuracy, precision, recall, f1, auc, cm


def train_with_gradual_unfreezing(model_class='improved', num_epochs=30, use_focal_loss=True, 
                                  use_mixup=True, use_tta=False, save_path=None):
    # Create dataloaders
    train_loader, val_loader, test_loader, class_weights = create_improved_dataloaders(
        DATA_DIR, SPLITS_DIR, NUM_FRAMES, IMG_SIZE, BATCH_SIZE, num_workers=2, use_augmentation=True
    )
    
    # Initialize model
    if model_class == 'improved':
        model = ImprovedResNet3D(num_classes=2, pretrained=True, dropout=0.5, use_attention=True)
    elif model_class == 'multiscale':
        model = MultiScaleResNet3D(num_classes=2, pretrained=True, dropout=0.5)
    elif model_class == 'auxiliary':
        model = ResNet3DWithAuxiliary(num_classes=2, pretrained=True, dropout=0.5)
    else:
        raise ValueError(f"Unknown model class: {model_class}")
    
    model = model.to(DEVICE)
    
    # Initialize loss function
    if use_focal_loss:
        criterion = FocalLoss(alpha=class_weights.to(DEVICE), gamma=2.0)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    
    # Phase 1: Train only classifier head (freeze backbone)
    print("\n" + "="*60)
    print("PHASE 1: Training classifier head only (backbone frozen)")
    print("="*60)
    
    if hasattr(model, 'freeze_backbone'):
        model.freeze_backbone()
    
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                           lr=LEARNING_RATE * 10, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    
    phase1_epochs = min(5, num_epochs // 3)
    best_val_acc = 0.0
    best_model_state = None
    
    for epoch in range(phase1_epochs):
        print(f"\nEpoch {epoch+1}/{phase1_epochs}")
        print("-" * 50)
        
        if model_class == 'auxiliary':
            train_loss, train_acc, train_precision, train_recall, train_f1 = \
                train_epoch_auxiliary(model, train_loader, criterion, optimizer, DEVICE)
            print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, "
                  f"Prec: {train_precision:.2f}, Rec: {train_recall:.2f}, F1: {train_f1:.2f}")
        else:
            train_loss, train_acc, train_precision, train_recall, train_f1, train_auc = \
                train_epoch_basic(model, train_loader, criterion, optimizer, DEVICE, use_mixup=False)
            print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, "
                  f"Prec: {train_precision:.2f}, Rec: {train_recall:.2f}, F1: {train_f1:.2f}, AUC: {train_auc:.2f}")
        
        val_loss, val_acc, val_precision, val_recall, val_f1, val_auc, cm = \
            validate(model, val_loader, criterion, DEVICE, use_tta=False)
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, "
              f"Prec: {val_precision:.2f}, Rec: {val_recall:.2f}, F1: {val_f1:.2f}, AUC: {val_auc:.2f}")
        
        scheduler.step()
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
    
    # Phase 2: Unfreeze and fine-tune entire model
    print("\n" + "="*60)
    print("PHASE 2: Fine-tuning entire model")
    print("="*60)
    
    if hasattr(model, 'unfreeze_backbone'):
        model.unfreeze_backbone()
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, 
                                                     patience=3)
    early_stopping = EarlyStopping(patience=7, mode='max')
    
    phase2_epochs = num_epochs - phase1_epochs
    
    for epoch in range(phase2_epochs):
        print(f"\nEpoch {phase1_epochs + epoch + 1}/{num_epochs}")
        print("-" * 50)
        
        if model_class == 'auxiliary':
            train_loss, train_acc, train_precision, train_recall, train_f1 = \
                train_epoch_auxiliary(model, train_loader, criterion, optimizer, DEVICE)
            print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, "
                  f"Prec: {train_precision:.2f}, Rec: {train_recall:.2f}, F1: {train_f1:.2f}")
        else:
            train_loss, train_acc, train_precision, train_recall, train_f1, train_auc = \
                train_epoch_basic(model, train_loader, criterion, optimizer, DEVICE, use_mixup=use_mixup)
            print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, "
                  f"Prec: {train_precision:.2f}, Rec: {train_recall:.2f}, F1: {train_f1:.2f}, AUC: {train_auc:.2f}")
        
        val_loss, val_acc, val_precision, val_recall, val_f1, val_auc, cm = \
            validate(model, val_loader, criterion, DEVICE, use_tta=use_tta)
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, "
              f"Prec: {val_precision:.2f}, Rec: {val_recall:.2f}, F1: {val_f1:.2f}, AUC: {val_auc:.2f}")
        print(f"Confusion Matrix:\n{cm}")
        
        scheduler.step(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            if save_path:
                torch.save(best_model_state, save_path)
                print(f"Saved best model with val acc: {val_acc:.2f}%")
        
        if early_stopping(val_acc):
            print(f"Early stopping triggered at epoch {phase1_epochs + epoch + 1}")
            break
    
    # Load best model and evaluate on test set
    print("\n" + "="*60)
    print("FINAL EVALUATION ON TEST SET")
    print("="*60)
    
    model.load_state_dict(best_model_state)
    test_loss, test_acc, test_precision, test_recall, test_f1, test_auc, test_cm = \
        validate(model, test_loader, criterion, DEVICE, use_tta=use_tta)
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Test Precision: {test_precision:.2f}")
    print(f"Test Recall: {test_recall:.2f}")
    print(f"Test F1: {test_f1:.2f}")
    print(f"Test AUC: {test_auc:.2f}")
    print(f"Test Confusion Matrix:\n{test_cm}")
    
    return model, best_val_acc, test_acc


if __name__ == '__main__':
    # Train improved model with attention
    print("Training ImprovedResNet3D with attention mechanisms...")
    model1, val_acc1, test_acc1 = train_with_gradual_unfreezing(
        model_class='improved',
        num_epochs=30,
        use_focal_loss=True,
        use_mixup=True,
        use_tta=True,
        save_path='best_improved_resnet3d.pth'
    )
    
    print("\n" + "="*60)
    print(f"Final Results - ImprovedResNet3D:")
    print(f"Best Val Accuracy: {val_acc1:.2f}%")
    print(f"Test Accuracy: {test_acc1:.2f}%")
    print("="*60)
