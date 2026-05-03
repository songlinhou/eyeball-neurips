"""
Multi-class video dataset for hierarchical diagnosis classification.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os


class MultiClassVideoDataset(Dataset):
    """
    Dataset for multi-class video classification with hierarchical labels.
    
    Each sample has:
    - diagnostic_class: Primary diagnosis (non_rd=0, rd=1)
    - subtype: Subtype classification (macula_detached, macula_intact, normal, pvd)
    """
    
    def __init__(self, split_file, data_root, num_frames=32, img_size=224, 
                 transform=None, is_training=False):
        """
        Args:
            split_file: Path to split file (train.txt or test.txt)
            data_root: Root directory containing video files
            num_frames: Number of frames to sample from each video
            img_size: Size to resize frames to
            transform: Optional albumentations transform
            is_training: Whether this is training set (for augmentation)
        """
        self.data_root = Path(data_root)
        self.num_frames = num_frames
        self.img_size = img_size
        self.is_training = is_training
        
        # Load split file
        self.samples = []
        with open(split_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) == 3:
                    video_path, diagnostic_label, subtype_label = parts
                    self.samples.append({
                        'video_path': video_path,
                        'diagnostic_label': int(diagnostic_label),
                        'subtype_label': int(subtype_label)
                    })
        
        # Setup transforms
        if transform is not None:
            self.transform = transform
        elif is_training:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.GaussNoise(p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
    
    def __len__(self):
        return len(self.samples)
    
    def load_video(self, video_path):
        """Load and sample frames from video"""
        full_path = self.data_root / video_path
        
        if not full_path.exists():
            raise FileNotFoundError(f"Video not found: {full_path}")
        
        cap = cv2.VideoCapture(str(full_path))
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        cap.release()
        
        if len(frames) == 0:
            raise ValueError(f"No frames loaded from {full_path}")
        
        # Sample frames uniformly
        if len(frames) >= self.num_frames:
            indices = np.linspace(0, len(frames) - 1, self.num_frames, dtype=int)
        else:
            # Repeat frames if video is too short
            indices = np.linspace(0, len(frames) - 1, self.num_frames, dtype=int)
        
        sampled_frames = [frames[i] for i in indices]
        
        return sampled_frames
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load video frames
        frames = self.load_video(sample['video_path'])
        
        # Apply transforms to each frame
        transformed_frames = []
        for frame in frames:
            transformed = self.transform(image=frame)
            transformed_frames.append(transformed['image'])
        
        # Stack frames: (T, C, H, W)
        video_tensor = torch.stack(transformed_frames)
        
        # Rearrange to (C, T, H, W) for 3D CNNs
        video_tensor = video_tensor.permute(1, 0, 2, 3)
        
        # Return video, diagnostic label, subtype label
        return (
            video_tensor,
            torch.tensor(sample['diagnostic_label'], dtype=torch.long),
            torch.tensor(sample['subtype_label'], dtype=torch.long)
        )


def create_multiclass_dataloaders(
    data_root,
    splits_dir,
    num_frames=32,
    img_size=224,
    batch_size=8,
    num_workers=4,
    use_augmentation=True
):
    """
    Create train and test dataloaders for multi-class classification.
    
    Args:
        data_root: Root directory containing video files
        splits_dir: Directory containing split files
        num_frames: Number of frames per video
        img_size: Image size
        batch_size: Batch size
        num_workers: Number of data loading workers
        use_augmentation: Whether to use data augmentation for training
    
    Returns:
        train_loader, test_loader, class_weights_diagnostic, class_weights_subtype
    """
    
    train_file = os.path.join(splits_dir, 'train.txt')
    test_file = os.path.join(splits_dir, 'test.txt')
    
    # Create datasets
    train_dataset = MultiClassVideoDataset(
        train_file, data_root, num_frames, img_size,
        is_training=use_augmentation
    )
    
    test_dataset = MultiClassVideoDataset(
        test_file, data_root, num_frames, img_size,
        is_training=False
    )
    
    # Calculate class weights for both tasks
    diagnostic_labels = [s['diagnostic_label'] for s in train_dataset.samples]
    subtype_labels = [s['subtype_label'] for s in train_dataset.samples]
    
    # Diagnostic class weights
    diagnostic_counts = np.bincount(diagnostic_labels)
    diagnostic_weights = 1.0 / diagnostic_counts
    diagnostic_weights = diagnostic_weights / diagnostic_weights.sum()
    diagnostic_weights = torch.FloatTensor(diagnostic_weights)
    
    # Subtype class weights
    subtype_counts = np.bincount(subtype_labels)
    subtype_weights = 1.0 / subtype_counts
    subtype_weights = subtype_weights / subtype_weights.sum()
    subtype_weights = torch.FloatTensor(subtype_weights)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader, diagnostic_weights, subtype_weights
