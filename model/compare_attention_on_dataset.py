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


def analyze_attention_statistics(model, dataloader, num_samples=50):
    """
    Analyze attention statistics across dataset
    
    Args:
        model: Trained model
        dataloader: DataLoader for ERDES dataset
        num_samples: Number of samples to analyze
    """
    import numpy as np
    
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
    
    print("\nAnalyzing attention statistics across dataset...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_samples:
                break
            
            # Extract video tensor (adjust based on your dataloader format)
            if isinstance(batch, dict):
                video = batch['video'].to(device)
            elif isinstance(batch, (list, tuple)):
                video = batch[0].to(device)
            else:
                video = batch.to(device)
            
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
        # You'll need to implement your dataloader here
        print("Please implement dataloader for your dataset format")
        # Example:
        # from your_dataset import create_dataloader
        # dataloader = create_dataloader(args.data_dir, batch_size=4)
        # cbam_stats, spatial_stats = analyze_attention_statistics(model, dataloader, args.num_samples)
    
    else:
        print("Please provide either --video_paths or --data_dir")
