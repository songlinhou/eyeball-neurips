#!/usr/bin/env python3
"""
Visualize Important Frames and Attention Heatmaps from Multiclass Model

This script loads a trained multiclass model and visualizes:
1. Important frames selected by frame attention
2. Spatial attention heatmaps overlaid on frames
3. Frame importance scores over time
"""

import os
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import cv2
from PIL import Image

from multiclass_model import create_multiclass_model
from erdes_dataset import ERDESDataset


def denormalize_frame(frame):
    """
    Denormalize frame from ImageNet normalization
    
    Args:
        frame: Tensor (C, H, W) with ImageNet normalization
        
    Returns:
        frame: Tensor (C, H, W) in [0, 1] range
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    frame = frame * std + mean
    frame = torch.clamp(frame, 0, 1)
    
    return frame


def apply_heatmap_overlay(frame, attention_map, alpha=0.5, colormap='jet'):
    """
    Apply attention heatmap overlay on frame
    
    Args:
        frame: numpy array (H, W, 3) in [0, 1] range
        attention_map: numpy array (H, W) in [0, 1] range
        alpha: Overlay transparency
        colormap: Matplotlib colormap name
        
    Returns:
        overlay: numpy array (H, W, 3) with heatmap overlay
    """
    # Apply colormap to attention
    cmap = plt.get_cmap(colormap)
    heatmap = cmap(attention_map)[:, :, :3]  # (H, W, 3)
    
    # Blend frame and heatmap
    overlay = (1 - alpha) * frame + alpha * heatmap
    overlay = np.clip(overlay, 0, 1)
    
    return overlay


def visualize_sample(model, video, labels, metadata, device='cuda', 
                     output_path=None, top_k=5):
    """
    Visualize important frames and attention for a single sample
    
    Args:
        model: Trained multiclass model
        video: Video tensor (C, T, H, W)
        labels: Label dict
        metadata: Metadata dict
        device: Device to use
        output_path: Path to save visualization
        top_k: Number of important frames to visualize
    """
    model.eval()
    
    # Add batch dimension
    video = video.unsqueeze(0).to(device)  # (1, C, T, H, W)
    
    with torch.no_grad():
        # Forward pass
        outputs = model(video)
        
        # Get predictions
        diagnostic_pred = outputs['diagnostic'].argmax(dim=1).item()
        subtype_pred = outputs['subtype'].argmax(dim=1).item()
        
        diagnostic_conf = F.softmax(outputs['diagnostic'], dim=1)[0, diagnostic_pred].item()
        subtype_conf = F.softmax(outputs['subtype'], dim=1)[0, subtype_pred].item()
        
        # Extract important frames and attention
        important_frames, frame_indices, importance_scores, important_attention = \
            model.get_important_frames(video, top_k=top_k)
        
        # Move to CPU
        important_frames = important_frames[0].cpu()  # (top_k, C, H, W)
        frame_indices = frame_indices[0].cpu().numpy()  # (top_k,)
        importance_scores = importance_scores[0].cpu().numpy()  # (T,)
        important_attention = important_attention[0].cpu()  # (top_k, 1, H, W)
    
    # Create visualization
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, top_k, figure=fig, hspace=0.3, wspace=0.2)
    
    # Title
    clip_id = metadata['clip_id']
    gt_diagnostic = metadata['diagnostic_class']
    gt_subtype = metadata['subtype']
    
    diagnostic_names = ['non_rd', 'rd']
    subtype_names = ['normal', 'macula_intact', 'macula_detached', 'pvd']
    
    pred_diagnostic = diagnostic_names[diagnostic_pred]
    pred_subtype = subtype_names[subtype_pred]
    
    fig.suptitle(
        f'Clip: {clip_id}\n'
        f'Ground Truth: {gt_diagnostic} / {gt_subtype}\n'
        f'Prediction: {pred_diagnostic} ({diagnostic_conf:.2%}) / {pred_subtype} ({subtype_conf:.2%})',
        fontsize=14, fontweight='bold'
    )
    
    # Row 1: Original important frames
    for i in range(top_k):
        ax = fig.add_subplot(gs[0, i])
        
        # Denormalize and convert to numpy
        frame = denormalize_frame(important_frames[i])  # (C, H, W)
        frame_np = frame.permute(1, 2, 0).numpy()  # (H, W, C)
        
        ax.imshow(frame_np)
        ax.set_title(f'Frame {frame_indices[i]}\nScore: {importance_scores[frame_indices[i]]:.3f}', 
                     fontsize=10)
        ax.axis('off')
    
    # Row 2: Attention heatmaps
    for i in range(top_k):
        ax = fig.add_subplot(gs[1, i])
        
        # Get attention map
        attention = important_attention[i, 0].numpy()  # (H, W)
        
        # Normalize to [0, 1]
        attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)
        
        im = ax.imshow(attention, cmap='jet')
        ax.set_title(f'Attention Map', fontsize=10)
        ax.axis('off')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Row 3: Heatmap overlays
    for i in range(top_k):
        ax = fig.add_subplot(gs[2, i])
        
        # Denormalize frame
        frame = denormalize_frame(important_frames[i])
        frame_np = frame.permute(1, 2, 0).numpy()
        
        # Get attention map
        attention = important_attention[i, 0].numpy()
        attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)
        
        # Apply overlay
        overlay = apply_heatmap_overlay(frame_np, attention, alpha=0.5)
        
        ax.imshow(overlay)
        ax.set_title(f'Overlay', fontsize=10)
        ax.axis('off')
    
    # Add frame importance timeline at the bottom
    fig.add_subplot(gs[2, :])
    plt.clf()
    ax_timeline = fig.add_subplot(gs[2, :])
    
    T = len(importance_scores)
    ax_timeline.plot(range(T), importance_scores, 'b-', linewidth=2, label='Frame Importance')
    ax_timeline.scatter(frame_indices, importance_scores[frame_indices], 
                       c='red', s=100, zorder=5, label=f'Top-{top_k} Frames')
    
    ax_timeline.set_xlabel('Frame Index', fontsize=12)
    ax_timeline.set_ylabel('Importance Score', fontsize=12)
    ax_timeline.set_title('Frame Importance Over Time', fontsize=12, fontweight='bold')
    ax_timeline.grid(True, alpha=0.3)
    ax_timeline.legend()
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_batch(model, dataloader, device='cuda', output_dir='./visualizations',
                   num_samples=5, top_k=5):
    """
    Visualize multiple samples from dataloader
    
    Args:
        model: Trained multiclass model
        dataloader: DataLoader
        device: Device to use
        output_dir: Directory to save visualizations
        num_samples: Number of samples to visualize
        top_k: Number of important frames per sample
    """
    os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    
    count = 0
    for videos, labels, metadata_list in dataloader:
        for i in range(videos.size(0)):
            if count >= num_samples:
                return
            
            video = videos[i]
            label = {k: v[i] for k, v in labels.items()}
            metadata = metadata_list[i]
            
            output_path = os.path.join(output_dir, f"{metadata['clip_id']}_attention.png")
            
            print(f"\nVisualizing sample {count + 1}/{num_samples}: {metadata['clip_id']}")
            visualize_sample(model, video, label, metadata, device, output_path, top_k)
            
            count += 1


def create_comparison_grid(model, dataloader, device='cuda', output_path='comparison_grid.png',
                           num_samples=4, top_k=3):
    """
    Create a comparison grid showing multiple samples side-by-side
    
    Args:
        model: Trained multiclass model
        dataloader: DataLoader
        device: Device to use
        output_path: Path to save grid
        num_samples: Number of samples to compare
        top_k: Number of frames per sample
    """
    model.eval()
    
    samples_data = []
    
    # Collect samples
    for videos, labels, metadata_list in dataloader:
        for i in range(videos.size(0)):
            if len(samples_data) >= num_samples:
                break
            
            video = videos[i].unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(video)
                important_frames, frame_indices, importance_scores, important_attention = \
                    model.get_important_frames(video, top_k=top_k)
            
            samples_data.append({
                'clip_id': metadata_list[i]['clip_id'],
                'gt_diagnostic': metadata_list[i]['diagnostic_class'],
                'gt_subtype': metadata_list[i]['subtype'],
                'pred_diagnostic': outputs['diagnostic'].argmax(dim=1).item(),
                'pred_subtype': outputs['subtype'].argmax(dim=1).item(),
                'frames': important_frames[0].cpu(),
                'attention': important_attention[0].cpu(),
                'indices': frame_indices[0].cpu().numpy()
            })
        
        if len(samples_data) >= num_samples:
            break
    
    # Create grid
    fig, axes = plt.subplots(num_samples, top_k, figsize=(top_k * 4, num_samples * 4))
    
    diagnostic_names = ['non_rd', 'rd']
    subtype_names = ['normal', 'macula_intact', 'macula_detached', 'pvd']
    
    for row, sample in enumerate(samples_data):
        for col in range(top_k):
            ax = axes[row, col] if num_samples > 1 else axes[col]
            
            # Get frame and attention
            frame = denormalize_frame(sample['frames'][col])
            frame_np = frame.permute(1, 2, 0).numpy()
            
            attention = sample['attention'][col, 0].numpy()
            attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)
            
            # Create overlay
            overlay = apply_heatmap_overlay(frame_np, attention, alpha=0.5)
            
            ax.imshow(overlay)
            
            # Title for first column
            if col == 0:
                pred_diag = diagnostic_names[sample['pred_diagnostic']]
                pred_sub = subtype_names[sample['pred_subtype']]
                ax.set_ylabel(
                    f"{sample['clip_id']}\n"
                    f"GT: {sample['gt_diagnostic']}/{sample['gt_subtype']}\n"
                    f"Pred: {pred_diag}/{pred_sub}",
                    fontsize=9
                )
            
            ax.set_title(f"Frame {sample['indices'][col]}", fontsize=9)
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved comparison grid to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize attention from multiclass model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--csv_path', type=str,
                       default='../benchmarks/input/balanced_split_desc.csv',
                       help='Path to CSV file')
    parser.add_argument('--data_root', type=str, default='../erdes',
                       help='Root directory for video data')
    parser.add_argument('--output_dir', type=str, default='./visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples to visualize')
    parser.add_argument('--top_k', type=int, default=5,
                       help='Number of important frames to show')
    parser.add_argument('--num_frames', type=int, default=32,
                       help='Number of frames to sample from video')
    parser.add_argument('--img_size', type=int, default=224,
                       help='Image size')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--comparison_grid', action='store_true',
                       help='Create comparison grid instead of individual visualizations')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    model = create_multiclass_model(
        num_diagnostic_classes=2,
        num_subtype_classes=4,
        pretrained=False,
        dropout=0.3
    )
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    print("Model loaded successfully!")
    
    # Load dataset
    print(f"\nLoading dataset from {args.csv_path}...")
    dataset = ERDESDataset(
        csv_path=args.csv_path,
        data_root=args.data_root,
        num_frames=args.num_frames,
        img_size=args.img_size,
        split='test',
        use_augmentation=False
    )
    
    from torch.utils.data import DataLoader
    from erdes_dataset import collate_fn
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn
    )
    
    print(f"Dataset loaded: {len(dataset)} samples")
    
    # Create visualizations
    if args.comparison_grid:
        print("\nCreating comparison grid...")
        output_path = os.path.join(args.output_dir, 'comparison_grid.png')
        os.makedirs(args.output_dir, exist_ok=True)
        create_comparison_grid(
            model, dataloader, device, output_path,
            num_samples=args.num_samples, top_k=args.top_k
        )
    else:
        print(f"\nCreating individual visualizations for {args.num_samples} samples...")
        visualize_batch(
            model, dataloader, device, args.output_dir,
            num_samples=args.num_samples, top_k=args.top_k
        )
    
    print("\nVisualization complete!")


if __name__ == "__main__":
    main()
