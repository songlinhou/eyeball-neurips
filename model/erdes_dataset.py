"""
ERDES Dataset Loader for VLM Training
Reads from balanced_split_desc.csv and loads videos with ground truth labels
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional
import json


class ERDESDataset(Dataset):
    """
    ERDES Dataset for multi-class video classification
    
    Reads from balanced_split_desc.csv with structure:
    - diagnostic_class: non_rd, rd
    - subtype: normal, macula_on, macula_off, pvd
    - anatomical_subclass: superior, inferior, temporal, nasal, multiple (or nan for normal)
    """
    
    def __init__(self,
                 csv_path: str,
                 data_root: str = "../erdes",
                 num_frames: int = 32,
                 img_size: int = 224,
                 split: str = 'train',
                 transform=None):
        """
        Args:
            csv_path: Path to balanced_split_desc.csv
            data_root: Root directory containing the clips
            num_frames: Number of frames to sample
            img_size: Image size (will be resized to img_size x img_size)
            split: 'train', 'val', or 'test'
            transform: Optional transform
        """
        self.csv_path = csv_path
        self.data_root = Path(data_root)
        self.num_frames = num_frames
        self.img_size = img_size
        self.split = split
        self.transform = transform
        
        # Load CSV
        self.df = pd.read_csv(csv_path)
        
        # Create label mappings
        self.diagnostic_to_idx = {
            'non_rd': 0,
            'rd': 1
        }
        
        self.subtype_to_idx = {
            'normal': 0,
            'macula_intact': 1,
            'macula_detached': 2,
            'pvd': 3
        }
        
        self.anatomical_to_idx = {
            'N/A': 0,         # Not applicable (for normal/pvd cases)
            'n/a': 0,         # Lowercase variant
            'TD': 1,          # Total Detachment
            'ND': 2,          # Nasal Detachment
            'Bilateral': 3,   # Bilateral detachment
            'SD': 4,          # Superior Detachment
            'ID': 5           # Inferior Detachment
        }
        
        # Reverse mappings
        self.idx_to_diagnostic = {v: k for k, v in self.diagnostic_to_idx.items()}
        self.idx_to_subtype = {v: k for k, v in self.subtype_to_idx.items()}
        self.idx_to_anatomical = {v: k for k, v in self.anatomical_to_idx.items()}
        
        print(f"Loaded {len(self.df)} samples from {csv_path}")
        print(f"Diagnostic classes: {self.df['diagnostic_class'].value_counts().to_dict()}")
        print(f"Subtype classes: {self.df['subtype'].value_counts().to_dict()}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, Dict, str]:
        """
        Returns:
            video: Tensor of shape (C, T, H, W)
            labels: Dict with 'diagnostic', 'subtype', 'anatomical' indices
            metadata: Dict with video info and ground truth text
        """
        row = self.df.iloc[idx]
        
        # Get video path
        video_path = self.data_root / row['file_path']
        
        # Load video
        video = self._load_video(video_path)
        
        # Get labels (only diagnostic and subtype, not anatomical)
        diagnostic_label = self.diagnostic_to_idx[row['diagnostic_class']]
        subtype_label = self.subtype_to_idx[row['subtype']]
        
        labels = {
            'diagnostic': diagnostic_label,
            'subtype': subtype_label
        }
        
        # Metadata (keep anatomical_subclass for reference but don't use for training)
        anatomical_str = row['anatomical_subclass']
        if pd.isna(anatomical_str) or anatomical_str == '' or anatomical_str == 'nan':
            anatomical_str = 'N/A'
        else:
            anatomical_str = str(anatomical_str).strip()
        
        metadata = {
            'clip_id': row['clip_id'],
            'file_path': row['file_path'],
            'diagnostic_class': row['diagnostic_class'],
            'subtype': row['subtype'],
            'anatomical_subclass': anatomical_str,  # For reference only
            'fps': row['fps'],
            'frame_count': row['frame_count'],
            'duration_seconds': row['duration_seconds']
        }
        
        return video, labels, metadata
    
    def _load_video(self, video_path: Path) -> torch.Tensor:
        """
        Load video and sample frames
        
        Returns:
            video: Tensor of shape (C, T, H, W)
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize
            frame = cv2.resize(frame, (self.img_size, self.img_size))
            
            frames.append(frame)
        
        cap.release()
        
        if len(frames) == 0:
            raise ValueError(f"No frames loaded from {video_path}")
        
        # Sample num_frames uniformly
        frames = np.array(frames)  # (T, H, W, C)
        total_frames = len(frames)
        
        if total_frames >= self.num_frames:
            # Uniform sampling
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            frames = frames[indices]
        else:
            # Repeat frames if video is too short
            repeat_factor = int(np.ceil(self.num_frames / total_frames))
            frames = np.tile(frames, (repeat_factor, 1, 1, 1))[:self.num_frames]
        
        # Convert to tensor: (T, H, W, C) -> (C, T, H, W)
        video = torch.from_numpy(frames).float()
        video = video.permute(3, 0, 1, 2)  # (C, T, H, W)
        
        # Normalize to [0, 1]
        video = video / 255.0
        
        # Apply ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
        video = (video - mean) / std
        
        return video


def create_erdes_dataloaders(csv_path: str,
                             data_root: str = "../erdes",
                             num_frames: int = 32,
                             img_size: int = 224,
                             batch_size: int = 16,
                             num_workers: int = 4,
                             train_split: float = 0.7,
                             val_split: float = 0.15):
    """
    Create train/val/test dataloaders from ERDES CSV
    
    Args:
        csv_path: Path to balanced_split_desc.csv
        data_root: Root directory
        num_frames: Number of frames to sample
        img_size: Image size
        batch_size: Batch size
        num_workers: Number of workers
        train_split: Training split ratio
        val_split: Validation split ratio
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Load full dataset
    full_dataset = ERDESDataset(
        csv_path=csv_path,
        data_root=data_root,
        num_frames=num_frames,
        img_size=img_size
    )
    
    # Split dataset
    total_size = len(full_dataset)
    train_size = int(total_size * train_split)
    val_size = int(total_size * val_split)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\nDataset splits:")
    print(f"  Train: {train_size} samples")
    print(f"  Val: {val_size} samples")
    print(f"  Test: {test_size} samples")
    
    return train_loader, val_loader, test_loader


def collate_fn(batch):
    """
    Custom collate function for ERDES dataset
    
    Args:
        batch: List of (video, labels, metadata) tuples
        
    Returns:
        videos: Batched videos (B, C, T, H, W)
        labels: Dict of batched labels
        metadata: List of metadata dicts
    """
    videos = []
    diagnostic_labels = []
    subtype_labels = []
    metadata_list = []
    
    for video, labels, metadata in batch:
        videos.append(video)
        diagnostic_labels.append(labels['diagnostic'])
        subtype_labels.append(labels['subtype'])
        metadata_list.append(metadata)
    
    videos = torch.stack(videos)
    
    labels_dict = {
        'diagnostic': torch.tensor(diagnostic_labels, dtype=torch.long),
        'subtype': torch.tensor(subtype_labels, dtype=torch.long)
    }
    
    return videos, labels_dict, metadata_list


if __name__ == "__main__":
    # Test dataset loading
    csv_path = "../benchmarks/input/balanced_split_desc.csv"
    data_root = "../erdes"
    
    print("Testing ERDES Dataset...")
    dataset = ERDESDataset(csv_path, data_root, num_frames=32, img_size=224)
    
    print(f"\nLoading first sample...")
    video, labels, metadata = dataset[0]
    
    print(f"Video shape: {video.shape}")
    print(f"Labels: {labels}")
    print(f"Metadata keys: {list(metadata.keys())}")
    print(f"Clip ID: {metadata['clip_id']}")
    print(f"Diagnostic: {metadata['diagnostic_class']}")
    print(f"Subtype: {metadata['subtype']}")
    print(f"Anatomical: {metadata['anatomical_subclass']}")
    print(f"\nSummary: {metadata['summary'][:200]}...")
    
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_erdes_dataloaders(
        csv_path, data_root, batch_size=4
    )
    
    print("\nTesting batch loading...")
    for videos, labels, metadata_list in train_loader:
        print(f"Batch videos shape: {videos.shape}")
        print(f"Batch diagnostic labels: {labels['diagnostic']}")
        print(f"Batch subtype labels: {labels['subtype']}")
        print(f"Number of metadata entries: {len(metadata_list)}")
        print(f"Anatomical (metadata only): {[m['anatomical_subclass'] for m in metadata_list]}")
        break
    
    print("\nDataset test complete!")
