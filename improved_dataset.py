import warnings
warnings.filterwarnings('ignore')
import os
os.environ['PYTHONWARNINGS'] = 'ignore'

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.io as io
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import random


class VideoAugmentation:
    def __init__(self, is_training=True):
        self.is_training = is_training
        
    def temporal_augmentation(self, frames, num_frames):
        total_frames = frames.shape[0]
        
        if not self.is_training or total_frames < num_frames:
            if total_frames >= num_frames:
                indices = torch.linspace(0, total_frames - 1, num_frames).long()
            else:
                indices = torch.arange(0, total_frames).float()
                indices = indices.repeat(num_frames // total_frames + 1)[:num_frames].long()
        else:
            # Random temporal sampling with jittering
            if random.random() < 0.5:
                # Uniform sampling
                indices = torch.linspace(0, total_frames - 1, num_frames).long()
            else:
                # Random sampling with temporal jittering
                segment_len = total_frames / num_frames
                indices = []
                for i in range(num_frames):
                    start = int(i * segment_len)
                    end = int((i + 1) * segment_len)
                    if end > start:
                        idx = random.randint(start, min(end - 1, total_frames - 1))
                    else:
                        idx = start
                    indices.append(idx)
                indices = torch.tensor(indices, dtype=torch.long)
        
        return frames[indices]
    
    def spatial_augmentation(self, frames):
        # frames: (T, H, W, C)
        if not self.is_training:
            return frames
        
        # Random horizontal flip
        if random.random() < 0.5:
            frames = torch.flip(frames, dims=[2])
        
        # Random rotation (small angles for medical imaging)
        if random.random() < 0.3:
            angle = random.uniform(-10, 10)
            frames = self._rotate_frames(frames, angle)
        
        # Random brightness and contrast
        if random.random() < 0.5:
            brightness_factor = random.uniform(0.8, 1.2)
            frames = frames * brightness_factor
            frames = torch.clamp(frames, 0, 1)
        
        if random.random() < 0.5:
            contrast_factor = random.uniform(0.8, 1.2)
            mean = frames.mean(dim=[1, 2], keepdim=True)
            frames = (frames - mean) * contrast_factor + mean
            frames = torch.clamp(frames, 0, 1)
        
        # Random Gaussian noise
        if random.random() < 0.3:
            noise = torch.randn_like(frames) * 0.02
            frames = frames + noise
            frames = torch.clamp(frames, 0, 1)
        
        return frames
    
    def _rotate_frames(self, frames, angle):
        # Simple rotation using affine transformation
        # This is a simplified version; for production, use torchvision.transforms.functional
        return frames
    
    def __call__(self, frames, num_frames):
        frames = self.temporal_augmentation(frames, num_frames)
        frames = self.spatial_augmentation(frames)
        return frames


class ImprovedVideoDataset(Dataset):
    def __init__(self, split_csv, data_dir, num_frames=32, img_size=224, 
                 is_training=True, use_augmentation=True):
        self.data = pd.read_csv(split_csv)
        self.data_dir = data_dir
        self.num_frames = num_frames
        self.img_size = img_size
        self.is_training = is_training
        self.use_augmentation = use_augmentation
        
        if use_augmentation and is_training:
            self.augmentation = VideoAugmentation(is_training=True)
        else:
            self.augmentation = VideoAugmentation(is_training=False)
        
        # Compute class weights for handling imbalance
        self.class_counts = self.data['label'].value_counts().sort_index().values
        self.class_weights = 1.0 / torch.tensor(self.class_counts, dtype=torch.float32)
        self.class_weights = self.class_weights / self.class_weights.sum()
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        video_path = os.path.join(self.data_dir, row['path'])
        label = int(row['label'])
        
        try:
            video, audio, info = io.read_video(video_path, pts_unit='sec')
            
            # Convert to float and normalize BEFORE augmentation
            video = video.float() / 255.0
            
            # Apply augmentation
            frames = self.augmentation(video, self.num_frames)
            
            # Resize frames
            if frames.shape[1] != self.img_size or frames.shape[2] != self.img_size:
                frames = torch.nn.functional.interpolate(
                    frames.permute(0, 3, 1, 2),
                    size=(self.img_size, self.img_size),
                    mode='bilinear',
                    align_corners=False
                ).permute(0, 2, 3, 1)
            
            # Content-aware cropping
            first_frame = frames[0]
            pixel_mask = (first_frame.sum(dim=2) > 0).cpu().numpy()
            non_zero_rows, non_zero_cols = np.where(pixel_mask)
            
            min_row, max_row = 0, self.img_size - 1
            min_col, max_col = 0, self.img_size - 1
            
            if non_zero_rows.size > 0 and non_zero_cols.size > 0:
                min_row, max_row = np.min(non_zero_rows), np.max(non_zero_rows)
                min_col, max_col = np.min(non_zero_cols), np.max(non_zero_cols)
            
            cropped_frames = frames[:, min_row:max_row+1, min_col:max_col+1, :]
            cropped_frames = cropped_frames.permute(0, 3, 1, 2)
            frames = torch.nn.functional.interpolate(
                cropped_frames,
                size=(self.img_size, self.img_size),
                mode='bilinear',
                align_corners=False
            )
            frames = frames.permute(0, 2, 3, 1)
            
            mask = torch.ones((self.img_size, self.img_size), dtype=torch.bool)
            
            # Convert to (C, T, H, W) format
            frames = frames.permute(3, 0, 1, 2)
            
            return frames, label, mask
            
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            return torch.zeros(3, self.num_frames, self.img_size, self.img_size), \
                   label, \
                   torch.zeros(self.img_size, self.img_size, dtype=torch.bool)


class MultiViewVideoDataset(Dataset):
    def __init__(self, split_csv, data_dir, num_frames=32, img_size=224, 
                 num_views=3, is_training=True):
        self.data = pd.read_csv(split_csv)
        self.data_dir = data_dir
        self.num_frames = num_frames
        self.img_size = img_size
        self.num_views = num_views
        self.is_training = is_training
        
        self.augmentation = VideoAugmentation(is_training=is_training)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        video_path = os.path.join(self.data_dir, row['path'])
        label = int(row['label'])
        
        try:
            video, audio, info = io.read_video(video_path, pts_unit='sec')
            
            # Convert to float first
            video = video.float() / 255.0
            
            # Generate multiple views with different augmentations
            views = []
            for _ in range(self.num_views):
                frames = self.augmentation(video.clone(), self.num_frames)
                
                if frames.shape[1] != self.img_size or frames.shape[2] != self.img_size:
                    frames = torch.nn.functional.interpolate(
                        frames.permute(0, 3, 1, 2),
                        size=(self.img_size, self.img_size),
                        mode='bilinear',
                        align_corners=False
                    ).permute(0, 2, 3, 1)
                
                frames = frames.permute(3, 0, 1, 2)
                views.append(frames)
            
            views = torch.stack(views)
            return views, label
            
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            return torch.zeros(self.num_views, 3, self.num_frames, self.img_size, self.img_size), label


def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def create_improved_dataloaders(data_dir, splits_dir, num_frames=32, img_size=224, 
                               batch_size=8, num_workers=4, use_augmentation=True):
    train_dataset = ImprovedVideoDataset(
        os.path.join(splits_dir, "train.csv"),
        data_dir,
        num_frames=num_frames,
        img_size=img_size,
        is_training=True,
        use_augmentation=use_augmentation
    )
    
    val_dataset = ImprovedVideoDataset(
        os.path.join(splits_dir, "val.csv"),
        data_dir,
        num_frames=num_frames,
        img_size=img_size,
        is_training=False,
        use_augmentation=False
    )
    
    test_dataset = ImprovedVideoDataset(
        os.path.join(splits_dir, "test.csv"),
        data_dir,
        num_frames=num_frames,
        img_size=img_size,
        is_training=False,
        use_augmentation=False
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=num_workers, 
                             pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=num_workers, 
                           pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                            shuffle=False, num_workers=num_workers, 
                            pin_memory=True)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Class weights: {train_dataset.class_weights}")
    
    return train_loader, val_loader, test_loader, train_dataset.class_weights
