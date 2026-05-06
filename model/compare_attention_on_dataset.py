"""
Compare CBAM vs Spatial Explainability attention on actual ERDES dataset
"""

import torch
import sys
from pathlib import Path
from visualize_attention_comparison import MultiClassExplainableResNet3DWithCBAMAttention, visualize_attention_comparison, visualize_multiple_frames

# Add parent directory to path if needed
sys.path.append(str(Path(__file__).parent.parent))


def load_model(checkpoint_path, device='cuda'):
    """Load trained model from checkpoint"""
    model = MultiClassExplainableResNet3DWithCBAMAttention(
        num_diagnostic_classes=2,
        num_subtype_classes=4,
        pretrained=False,
        use_attention=True
    )
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model


def compare_attention_on_videos(model, video_paths, save_dir='attention_analysis'):
    """
    Compare attention for multiple videos from dataset
    
    Args:
        model: Trained model
        video_paths: List of paths to video files or preprocessed tensors
        save_dir: Directory to save results
    """
    from torch.utils.data import DataLoader
    
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    device = next(model.parameters()).device
    
    for idx, video_path in enumerate(video_paths):
        print(f"\n{'='*60}")
        print(f"Processing video {idx+1}/{len(video_paths)}: {video_path}")
        print(f"{'='*60}")
        
        # Load video tensor (adjust this based on your data format)
        if isinstance(video_path, str):
            video_path = Path(video_path)
            if video_path.suffix == '.pt':
                video_tensor = torch.load(video_path)
            else:
                # Load from video file - you'll need to implement this based on your preprocessing
                print(f"Skipping {video_path} - implement video loading for your format")
                continue
        else:
            # Assume it's already a tensor
            video_tensor = video_path
        
        # Ensure correct shape (B, C, T, H, W)
        if video_tensor.dim() == 4:
            video_tensor = video_tensor.unsqueeze(0)
        
        # Visualize single most important frame
        video_save_dir = save_dir / f'video_{idx}'
        video_save_dir.mkdir(exist_ok=True, parents=True)
        
        visualize_attention_comparison(
            model, 
            video_tensor, 
            frame_idx=None,
            save_dir=video_save_dir
        )
        
        # Visualize top 5 important frames
        visualize_multiple_frames(
            model,
            video_tensor,
            num_frames=5,
            save_dir=video_save_dir
        )


def analyze_attention_statistics(model, dataloader, num_samples=50, save_dir=None, num_visualize=10):
    """
    Analyze attention statistics across dataset
    
    Args:
        model: Trained model
        dataloader: DataLoader for ERDES dataset
        num_samples: Number of samples to analyze
        save_dir: Directory to save visualizations (if provided)
        num_visualize: Number of samples to visualize
    """
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    
    device = next(model.parameters()).device
    model.eval()
    
    cbam_stats = {
        'mean': [],
        'std': [],
        'max': [],
        'sparsity_50': [],  # % pixels > 0.5
        'sparsity_70': []   # % pixels > 0.7
    }
    
    spatial_stats = {
        'mean': [],
        'std': [],
        'max': [],
        'sparsity_50': [],
        'sparsity_70': []
    }
    
    # Create save directory if needed
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        visualizations_dir = save_dir / 'visualizations'
        visualizations_dir.mkdir(exist_ok=True, parents=True)
    
    print("\nAnalyzing attention statistics across dataset...")
    
    sample_count = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_samples:
                break
            
            # ERDESDataset returns (videos, labels_dict, metadata_list)
            videos, labels_dict, metadata_list = batch
            video = videos.to(device)
            
            # Get attention maps
            outputs, attention_dict = model(video, return_attention=True)
            
            frame_importance = attention_dict['frame_importance']  # (B, T)
            spatial_attention = attention_dict['spatial_attention']  # (B, 1, T, H, W)
            cbam_spatial_attention = attention_dict['cbam_spatial_attention']  # (B, 1, T, H, W)
            
            B, _, T, H, W = spatial_attention.shape
            
            # For each video in batch
            for b in range(B):
                # Get most important frame
                most_important_frame = torch.argmax(frame_importance[b]).item()
                
                # Get attention maps for that frame
                cbam_attn = cbam_spatial_attention[b, 0, most_important_frame, :, :].cpu().numpy()
                spatial_attn = spatial_attention[b, 0, most_important_frame, :, :].cpu().numpy()
                
                # Compute statistics
                cbam_stats['mean'].append(cbam_attn.mean())
                cbam_stats['std'].append(cbam_attn.std())
                cbam_stats['max'].append(cbam_attn.max())
                cbam_stats['sparsity_50'].append((cbam_attn > 0.5).sum() / cbam_attn.size * 100)
                cbam_stats['sparsity_70'].append((cbam_attn > 0.7).sum() / cbam_attn.size * 100)
                
                spatial_stats['mean'].append(spatial_attn.mean())
                spatial_stats['std'].append(spatial_attn.std())
                spatial_stats['max'].append(spatial_attn.max())
                spatial_stats['sparsity_50'].append((spatial_attn > 0.5).sum() / spatial_attn.size * 100)
                spatial_stats['sparsity_70'].append((spatial_attn > 0.7).sum() / spatial_attn.size * 100)
                
                # Save visualization for first num_visualize samples
                if save_dir and sample_count < num_visualize:
                    # Get metadata
                    metadata = metadata_list[b]
                    clip_id = metadata['clip_id']
                    diagnostic = metadata['diagnostic_class']
                    subtype = metadata['subtype']
                    
                    # Get original frame
                    original_frame = video[b, :, most_important_frame, :, :].cpu()
                    # Denormalize
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                    original_frame = original_frame * std + mean
                    original_frame = original_frame.permute(1, 2, 0).numpy()
                    original_frame = np.clip(original_frame, 0, 1)
                    
                    # Get target size from original frame
                    target_h, target_w = original_frame.shape[:2]
                    
                    # Upsample attention maps to match original frame size
                    cbam_attn_vis = cv2.resize(cbam_attn, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                    spatial_attn_vis = cv2.resize(spatial_attn, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                    
                    # Create visualization
                    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                    
                    # Row 1: CBAM
                    axes[0, 0].imshow(original_frame)
                    axes[0, 0].set_title('Original Frame', fontsize=12, fontweight='bold')
                    axes[0, 0].axis('off')
                    
                    axes[0, 1].imshow(cbam_attn_vis, cmap='jet', vmin=0, vmax=1)
                    axes[0, 1].set_title('CBAM Spatial Attention', fontsize=12, fontweight='bold')
                    axes[0, 1].axis('off')
                    plt.colorbar(axes[0, 1].images[0], ax=axes[0, 1], fraction=0.046)
                    
                    # Create CBAM overlay
                    heatmap_cbam = cv2.applyColorMap((cbam_attn_vis * 255).astype(np.uint8), cv2.COLORMAP_JET)
                    heatmap_cbam = cv2.cvtColor(heatmap_cbam, cv2.COLOR_BGR2RGB) / 255.0
                    overlay_cbam = 0.6 * original_frame + 0.4 * heatmap_cbam
                    axes[0, 2].imshow(overlay_cbam)
                    axes[0, 2].set_title('CBAM Overlay', fontsize=12, fontweight='bold')
                    axes[0, 2].axis('off')
                    
                    # Row 2: Spatial Explainability
                    axes[1, 0].imshow(original_frame)
                    axes[1, 0].set_title('Original Frame', fontsize=12, fontweight='bold')
                    axes[1, 0].axis('off')
                    
                    axes[1, 1].imshow(spatial_attn_vis, cmap='jet', vmin=0, vmax=1)
                    axes[1, 1].set_title('Spatial Explainability Attention', fontsize=12, fontweight='bold')
                    axes[1, 1].axis('off')
                    plt.colorbar(axes[1, 1].images[0], ax=axes[1, 1], fraction=0.046)
                    
                    # Create Spatial Explainability overlay
                    heatmap_spatial = cv2.applyColorMap((spatial_attn_vis * 255).astype(np.uint8), cv2.COLORMAP_JET)
                    heatmap_spatial = cv2.cvtColor(heatmap_spatial, cv2.COLOR_BGR2RGB) / 255.0
                    overlay_spatial = 0.6 * original_frame + 0.4 * heatmap_spatial
                    axes[1, 2].imshow(overlay_spatial)
                    axes[1, 2].set_title('Spatial Explainability Overlay', fontsize=12, fontweight='bold')
                    axes[1, 2].axis('off')
                    
                    fig.suptitle(f'Sample {sample_count}: {clip_id}\n'
                                f'Diagnostic: {diagnostic}, Subtype: {subtype}\n'
                                f'Frame {most_important_frame} (Importance: {frame_importance[b, most_important_frame]:.4f})',
                                fontsize=14, fontweight='bold')
                    
                    plt.tight_layout()
                    plt.savefig(visualizations_dir / f'sample_{sample_count:03d}_{clip_id}.png', 
                               dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    print(f"  Saved visualization {sample_count + 1}/{num_visualize}: {clip_id}")
                
                sample_count += 1
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_idx + 1} batches...")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("ATTENTION STATISTICS SUMMARY")
    print("="*80)
    
    print("\nCBAM Spatial Attention:")
    print(f"  Mean activation: {np.mean(cbam_stats['mean']):.4f} ± {np.std(cbam_stats['mean']):.4f}")
    print(f"  Std activation: {np.mean(cbam_stats['std']):.4f} ± {np.std(cbam_stats['std']):.4f}")
    print(f"  Max activation: {np.mean(cbam_stats['max']):.4f} ± {np.std(cbam_stats['max']):.4f}")
    print(f"  Sparsity (>0.5): {np.mean(cbam_stats['sparsity_50']):.2f}% ± {np.std(cbam_stats['sparsity_50']):.2f}%")
    print(f"  Sparsity (>0.7): {np.mean(cbam_stats['sparsity_70']):.2f}% ± {np.std(cbam_stats['sparsity_70']):.2f}%")
    
    print("\nSpatial Explainability Attention:")
    print(f"  Mean activation: {np.mean(spatial_stats['mean']):.4f} ± {np.std(spatial_stats['mean']):.4f}")
    print(f"  Std activation: {np.mean(spatial_stats['std']):.4f} ± {np.std(spatial_stats['std']):.4f}")
    print(f"  Max activation: {np.mean(spatial_stats['max']):.4f} ± {np.std(spatial_stats['max']):.4f}")
    print(f"  Sparsity (>0.5): {np.mean(spatial_stats['sparsity_50']):.2f}% ± {np.std(spatial_stats['sparsity_50']):.2f}%")
    print(f"  Sparsity (>0.7): {np.mean(spatial_stats['sparsity_70']):.2f}% ± {np.std(spatial_stats['sparsity_70']):.2f}%")
    
    print("\nComparison:")
    print(f"  CBAM is {'MORE' if np.mean(cbam_stats['sparsity_50']) < np.mean(spatial_stats['sparsity_50']) else 'LESS'} sparse than Spatial Explainability")
    print(f"  Difference in sparsity (>0.5): {abs(np.mean(cbam_stats['sparsity_50']) - np.mean(spatial_stats['sparsity_50'])):.2f}%")
    print("="*80)
    
    if save_dir:
        print(f"\nVisualizations saved to: {visualizations_dir}")
        print(f"Number of visualizations: {min(num_visualize, sample_count)}")
    
    return cbam_stats, spatial_stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare CBAM vs Spatial Explainability attention')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, help='Path to ERDES data directory')
    parser.add_argument('--video_paths', nargs='+', help='Specific video paths to visualize')
    parser.add_argument('--save_dir', type=str, default='attention_analysis', help='Save directory')
    parser.add_argument('--num_samples', type=int, default=50, help='Number of samples for statistics')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    # Load model
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = load_model(args.checkpoint, device=device)
    
    if args.video_paths:
        # Visualize specific videos
        compare_attention_on_videos(model, args.video_paths, save_dir=args.save_dir)
    
    elif args.data_dir:
        # Analyze statistics on dataset
        from erdes_dataset import ERDESDataset, collate_fn
        from torch.utils.data import DataLoader
        
        # Find CSV file
        csv_path = Path(args.data_dir).parent / 'benchmarks' / 'input' / 'balanced_split_desc.csv'
        if not csv_path.exists():
            # Try alternative location
            csv_path = Path(args.data_dir).parent / 'balanced_split_desc.csv'
        if not csv_path.exists():
            print(f"Error: Could not find balanced_split_desc.csv")
            print(f"Tried: {csv_path}")
            exit(1)
        
        print(f"Loading dataset from {csv_path}")
        print(f"Data root: {args.data_dir}")
        
        # Create dataset (use test split to avoid augmentation)
        dataset = ERDESDataset(
            csv_path=str(csv_path),
            data_root=args.data_dir,
            num_frames=32,
            img_size=224,
            split='test',
            use_augmentation=False
        )
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            num_workers=2,
            collate_fn=collate_fn
        )
        
        print(f"\nAnalyzing {args.num_samples} samples from dataset...")
        cbam_stats, spatial_stats = analyze_attention_statistics(
            model, 
            dataloader, 
            num_samples=args.num_samples,
            save_dir=args.save_dir,
            num_visualize=min(10, args.num_samples)  # Visualize up to 10 samples
        )
    
    else:
        print("Please provide either --video_paths or --data_dir")
