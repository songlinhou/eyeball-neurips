#!/usr/bin/env python3
"""
Quick Visualization Script - Interactive visualization of attention heatmaps

Usage:
    python quick_visualize.py --checkpoint path/to/model.pth --sample_idx 0
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from multiclass_model import create_multiclass_model
from erdes_dataset import ERDESDataset


def denormalize(frame):
    """Denormalize ImageNet normalized frame"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return torch.clamp(frame * std + mean, 0, 1)


def visualize_single_sample(checkpoint_path, csv_path, data_root, sample_idx=0, 
                           top_k=5, save_path=None):
    """
    Quick visualization of a single sample
    
    Args:
        checkpoint_path: Path to model checkpoint
        csv_path: Path to CSV file
        data_root: Root directory for videos
        sample_idx: Index of sample to visualize
        top_k: Number of important frames
        save_path: Optional path to save figure
    """
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model from {checkpoint_path}...")
    
    model = create_multiclass_model(num_diagnostic_classes=2, num_subtype_classes=4)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    
    # Load dataset
    print(f"Loading dataset...")
    dataset = ERDESDataset(
        csv_path=csv_path,
        data_root=data_root,
        num_frames=32,
        img_size=224,
        split='test',
        use_augmentation=False
    )
    
    # Get sample
    video, labels, metadata = dataset[sample_idx]
    video = video.unsqueeze(0).to(device)
    
    print(f"\nSample: {metadata['clip_id']}")
    print(f"Ground Truth: {metadata['diagnostic_class']} / {metadata['subtype']}")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(video)
        
        # Predictions
        diagnostic_pred = outputs['diagnostic'].argmax(dim=1).item()
        subtype_pred = outputs['subtype'].argmax(dim=1).item()
        diagnostic_conf = F.softmax(outputs['diagnostic'], dim=1)[0, diagnostic_pred].item()
        subtype_conf = F.softmax(outputs['subtype'], dim=1)[0, subtype_pred].item()
        
        # Get important frames
        important_frames, frame_indices, importance_scores, important_attention = \
            model.extract_important_frames(video, top_k=top_k)
    
    # Move to CPU
    important_frames = important_frames[0].cpu()
    frame_indices = frame_indices[0].cpu().numpy()
    importance_scores = importance_scores[0].cpu().numpy()
    important_attention = important_attention[0].cpu()
    
    # Map predictions
    diagnostic_names = ['non_rd', 'rd']
    subtype_names = ['normal', 'macula_intact', 'macula_detached', 'pvd']
    
    print(f"Prediction: {diagnostic_names[diagnostic_pred]} ({diagnostic_conf:.2%}) / "
          f"{subtype_names[subtype_pred]} ({subtype_conf:.2%})")
    print(f"\nTop-{top_k} important frame indices: {frame_indices}")
    
    # Use actual number of frames returned
    num_frames = len(important_frames)
    
    # Create visualization
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(3, num_frames + 1, figure=fig, hspace=0.3, wspace=0.3,
                  width_ratios=[1]*num_frames + [0.3])
    
    # Title
    fig.suptitle(
        f'Clip: {metadata["clip_id"]}\n'
        f'GT: {metadata["diagnostic_class"]} / {metadata["subtype"]} | '
        f'Pred: {diagnostic_names[diagnostic_pred]} ({diagnostic_conf:.1%}) / '
        f'{subtype_names[subtype_pred]} ({subtype_conf:.1%})',
        fontsize=14, fontweight='bold', y=0.98
    )
    
    # Visualize each important frame
    for i in range(num_frames):
        # Original frame
        ax1 = fig.add_subplot(gs[0, i])
        frame = denormalize(important_frames[i]).permute(1, 2, 0).numpy()
        ax1.imshow(frame)
        ax1.set_title(f'Frame {frame_indices[i]}\nScore: {importance_scores[frame_indices[i]]:.3f}')
        ax1.axis('off')
        
        # Attention heatmap
        ax2 = fig.add_subplot(gs[1, i])
        attention = important_attention[i, 0].numpy()
        attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)
        im = ax2.imshow(attention, cmap='jet')
        ax2.set_title('Attention Map')
        ax2.axis('off')
        
        # Overlay
        ax3 = fig.add_subplot(gs[2, i])
        # Resize attention to match frame size
        from scipy.ndimage import zoom
        h, w = frame.shape[:2]
        attention_resized = zoom(attention, (h / attention.shape[0], w / attention.shape[1]), order=1)
        heatmap = plt.get_cmap('jet')(attention_resized)[:, :, :3]
        overlay = 0.6 * frame + 0.4 * heatmap
        overlay = np.clip(overlay, 0, 1)
        ax3.imshow(overlay)
        ax3.set_title('Overlay')
        ax3.axis('off')
    
    # Add colorbar
    cbar_ax = fig.add_subplot(gs[1, -1])
    plt.colorbar(im, cax=cbar_ax, label='Attention')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved to {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    # Plot frame importance timeline
    fig, ax = plt.subplots(figsize=(12, 4))
    T = len(importance_scores)
    ax.plot(range(T), importance_scores, 'b-', linewidth=2, label='Frame Importance')
    ax.scatter(frame_indices, importance_scores[frame_indices], 
               c='red', s=150, zorder=5, marker='*', label=f'Top-{top_k} Frames')
    
    for idx in frame_indices:
        ax.axvline(x=idx, color='red', linestyle='--', alpha=0.3)
    
    ax.set_xlabel('Frame Index', fontsize=12)
    ax.set_ylabel('Importance Score', fontsize=12)
    ax.set_title(f'Frame Importance Timeline - {metadata["clip_id"]}', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        timeline_path = save_path.replace('.png', '_timeline.png')
        plt.savefig(timeline_path, dpi=150, bbox_inches='tight')
        print(f"Saved timeline to {timeline_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Quick visualization of attention')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--csv_path', type=str,
                       default='../benchmarks/input/balanced_split_desc.csv',
                       help='Path to CSV file')
    parser.add_argument('--data_root', type=str, default='../erdes',
                       help='Root directory for videos')
    parser.add_argument('--sample_idx', type=int, default=0,
                       help='Index of sample to visualize')
    parser.add_argument('--top_k', type=int, default=5,
                       help='Number of important frames')
    parser.add_argument('--save', type=str, default=None,
                       help='Path to save visualization (optional)')
    
    args = parser.parse_args()
    
    visualize_single_sample(
        checkpoint_path=args.checkpoint,
        csv_path=args.csv_path,
        data_root=args.data_root,
        sample_idx=args.sample_idx,
        top_k=args.top_k,
        save_path=args.save
    )


if __name__ == "__main__":
    main()
