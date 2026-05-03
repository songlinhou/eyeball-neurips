"""
Utility script to extract and visualize explainability features from trained models
"""
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import cv2
import os

from explainable_models import (
    ExplainableResNet3D,
    OpticalFlowResNet3D,
    ExplainableOpticalFlowResNet3D,
    TSMResNet3D
)
from improved_dataset import create_improved_dataloaders


def load_model(model_path, model_class, dropout=0.3, device='cuda'):
    """Load a trained model"""
    if model_class == 'explainable':
        model = ExplainableResNet3D(num_classes=2, pretrained=False, dropout=dropout)
    elif model_class == 'optical_flow':
        model = OpticalFlowResNet3D(num_classes=2, pretrained=False, dropout=dropout)
    elif model_class == 'explainable_flow':
        model = ExplainableOpticalFlowResNet3D(num_classes=2, pretrained=False, dropout=dropout)
    elif model_class == 'tsm':
        model = TSMResNet3D(num_classes=2, pretrained=False, dropout=dropout)
    else:
        raise ValueError(f"Unknown model class: {model_class}")
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    return model


def extract_attention_maps(model, video_tensor, device='cuda'):
    """
    Extract frame importance and spatial attention maps from explainable models
    
    Args:
        model: Trained explainable model
        video_tensor: Input video tensor (B, C, T, H, W)
        device: Device to run on
    
    Returns:
        dict with 'frame_importance' and 'spatial_attention' if available
    """
    model.eval()
    video_tensor = video_tensor.to(device)
    
    with torch.no_grad():
        if hasattr(model, 'forward') and 'return_attention' in model.forward.__code__.co_varnames:
            output, attention_maps = model(video_tensor, return_attention=True)
            return output, attention_maps
        else:
            output = model(video_tensor)
            return output, None


def visualize_frame_importance(frame_importance, save_path=None, title="Frame Importance"):
    """
    Visualize which frames are most important for classification
    
    Args:
        frame_importance: Tensor of shape (B, T) or (T,)
        save_path: Path to save the visualization
        title: Plot title
    """
    if len(frame_importance.shape) > 1:
        frame_importance = frame_importance[0]  # Take first sample
    
    frame_importance = frame_importance.cpu().numpy()
    num_frames = len(frame_importance)
    
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Bar plot
    bars = ax.bar(range(num_frames), frame_importance, color='steelblue', alpha=0.7)
    
    # Highlight top 5 most important frames
    top_indices = np.argsort(frame_importance)[-5:]
    for idx in top_indices:
        bars[idx].set_color('coral')
    
    ax.set_xlabel('Frame Index', fontsize=12)
    ax.set_ylabel('Importance Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add text for top frames
    for idx in top_indices:
        ax.text(idx, frame_importance[idx], f'{frame_importance[idx]:.3f}', 
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Frame importance visualization saved to: {save_path}")
    
    plt.close()
    
    return top_indices


def visualize_spatial_attention(spatial_attention, video_frames, save_path=None, 
                                top_frames=None, title="Spatial Attention"):
    """
    Visualize spatial attention maps overlaid on video frames
    
    Args:
        spatial_attention: Tensor of shape (B, 1, T, H, W)
        video_frames: Original video tensor (B, C, T, H, W)
        save_path: Path to save the visualization
        top_frames: Indices of frames to visualize (if None, show all or sample)
        title: Plot title
    """
    if len(spatial_attention.shape) == 5:
        spatial_attention = spatial_attention[0, 0]  # (T, H, W)
    
    if len(video_frames.shape) == 5:
        video_frames = video_frames[0]  # (C, T, H, W)
    
    spatial_attention = spatial_attention.cpu().numpy()
    video_frames = video_frames.cpu().numpy()
    
    # Transpose to (T, H, W, C) for visualization
    video_frames = np.transpose(video_frames, (1, 2, 3, 0))
    
    # Normalize video frames to [0, 1]
    video_frames = (video_frames - video_frames.min()) / (video_frames.max() - video_frames.min() + 1e-8)
    
    num_frames = spatial_attention.shape[0]
    
    # Select frames to visualize
    if top_frames is not None:
        frames_to_show = top_frames[:6]  # Show up to 6 frames
    elif num_frames > 8:
        # Sample frames evenly
        frames_to_show = np.linspace(0, num_frames-1, 8, dtype=int)
    else:
        frames_to_show = range(num_frames)
    
    num_show = len(frames_to_show)
    cols = min(4, num_show)
    rows = (num_show + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*3))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1 or cols == 1:
        axes = axes.reshape(rows, cols)
    
    for idx, frame_idx in enumerate(frames_to_show):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]
        
        # Get frame and attention
        frame = video_frames[frame_idx]
        attention = spatial_attention[frame_idx]
        
        # Resize attention to match frame size
        attention_resized = cv2.resize(attention, (frame.shape[1], frame.shape[0]))
        
        # Create heatmap overlay
        heatmap = plt.cm.jet(attention_resized)[:, :, :3]
        
        # Blend with original frame
        overlay = 0.6 * frame + 0.4 * heatmap
        overlay = np.clip(overlay, 0, 1)
        
        ax.imshow(overlay)
        ax.set_title(f'Frame {frame_idx}\nAttention: {attention.mean():.3f}', fontsize=10)
        ax.axis('off')
    
    # Hide empty subplots
    for idx in range(num_show, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Spatial attention visualization saved to: {save_path}")
    
    plt.close()


def generate_explainability_report(model, dataloader, save_dir, model_name, 
                                   num_samples=10, device='cuda'):
    """
    Generate comprehensive explainability report for a model
    
    Args:
        model: Trained explainable model
        dataloader: DataLoader with test samples
        save_dir: Directory to save visualizations
        model_name: Name of the model/experiment
        num_samples: Number of samples to visualize
        device: Device to run on
    """
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    sample_count = 0
    
    print(f"\nGenerating explainability report for {model_name}...")
    print(f"Saving to: {save_dir}")
    
    for batch_idx, (videos, labels, masks) in enumerate(dataloader):
        if sample_count >= num_samples:
            break
        
        videos = videos.to(device)
        labels = labels.to(device)
        
        # Extract attention maps
        outputs, attention_maps = extract_attention_maps(model, videos, device)
        
        if attention_maps is None:
            print(f"Model {model_name} does not support attention extraction")
            return
        
        # Process each sample in batch
        batch_size = videos.size(0)
        for i in range(min(batch_size, num_samples - sample_count)):
            sample_idx = sample_count + i
            
            # Get prediction
            pred_class = outputs[i].argmax().item()
            true_class = labels[i].item()
            confidence = torch.softmax(outputs[i], dim=0).max().item()
            
            class_names = ['Macula Intact', 'Macula Detached']
            
            # Create sample directory
            sample_dir = os.path.join(save_dir, f"sample_{sample_idx:03d}")
            os.makedirs(sample_dir, exist_ok=True)
            
            # Save sample info
            info_path = os.path.join(sample_dir, "info.txt")
            with open(info_path, 'w') as f:
                f.write(f"Sample {sample_idx}\n")
                f.write(f"True Label: {class_names[true_class]}\n")
                f.write(f"Predicted: {class_names[pred_class]}\n")
                f.write(f"Confidence: {confidence:.4f}\n")
                f.write(f"Correct: {pred_class == true_class}\n")
            
            # Visualize frame importance
            if 'frame_importance' in attention_maps:
                frame_imp = attention_maps['frame_importance'][i:i+1]
                frame_save_path = os.path.join(sample_dir, "frame_importance.png")
                title = f"Frame Importance - Sample {sample_idx}\nPred: {class_names[pred_class]} ({confidence:.2%})"
                top_frames = visualize_frame_importance(frame_imp, frame_save_path, title)
            else:
                top_frames = None
            
            # Visualize spatial attention
            if 'spatial_attention' in attention_maps:
                spatial_att = attention_maps['spatial_attention'][i:i+1]
                video = videos[i:i+1]
                spatial_save_path = os.path.join(sample_dir, "spatial_attention.png")
                title = f"Spatial Attention - Sample {sample_idx}\nTrue: {class_names[true_class]}, Pred: {class_names[pred_class]}"
                visualize_spatial_attention(spatial_att, video, spatial_save_path, top_frames, title)
            
            print(f"  Processed sample {sample_idx}: {class_names[true_class]} -> {class_names[pred_class]} ({confidence:.2%})")
        
        sample_count += batch_size
    
    print(f"\nExplainability report completed! Visualizations saved to: {save_dir}")


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize explainability features')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--model_class', type=str, required=True, 
                       choices=['explainable', 'optical_flow', 'explainable_flow', 'tsm'],
                       help='Model class')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save visualizations')
    parser.add_argument('--data_dir', type=str, default='../erdes', help='Data directory')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to visualize')
    parser.add_argument('--num_frames', type=int, default=32, help='Number of frames')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model = load_model(args.model_path, args.model_class, args.dropout, device)
    
    # Load data
    print("Loading data...")
    splits_dir = os.path.join(args.data_dir, "splits", "macula_detached_vs_intact")
    _, _, test_loader, _ = create_improved_dataloaders(
        args.data_dir, splits_dir,
        num_frames=args.num_frames,
        img_size=224,
        batch_size=args.batch_size,
        num_workers=2,
        use_augmentation=False
    )
    
    # Generate report
    model_name = Path(args.model_path).stem
    generate_explainability_report(
        model, test_loader, args.save_dir, model_name,
        num_samples=args.num_samples, device=device
    )


if __name__ == '__main__':
    main()
