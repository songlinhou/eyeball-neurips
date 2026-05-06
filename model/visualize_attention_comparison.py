"""
Visualize and compare CBAM spatial attention vs SpatialExplainabilityModule attention
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import cv2
from multiclass_model import MultiClassExplainableResNet3D


class MultiClassExplainableResNet3DWithCBAMAttention(MultiClassExplainableResNet3D):
    """Extended model that captures CBAM spatial attention for visualization"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cbam_spatial_attention = None
        
        # Hook to capture CBAM spatial attention
        if self.use_attention:
            self._register_cbam_hook()
    
    def _register_cbam_hook(self):
        """Register forward hook to capture CBAM spatial attention"""
        def hook_fn(module, input, output):
            # The SpatialAttention module computes attention and multiplies with input
            # We need to capture the attention map before multiplication
            x = input[0]
            avg_out = torch.mean(x, dim=1, keepdim=True)
            max_out, _ = torch.max(x, dim=1, keepdim=True)
            attention = torch.cat([avg_out, max_out], dim=1)
            attention = module.conv(attention)
            attention = module.sigmoid(attention)
            self.cbam_spatial_attention = attention.detach()
        
        # Register hook on the SpatialAttention module inside CBAM
        self.cbam.spatial_attention.register_forward_hook(hook_fn)
    
    def forward(self, x, return_attention=False):
        """Forward pass that captures both attention maps"""
        # Reset CBAM attention
        self.cbam_spatial_attention = None
        
        # Call parent forward
        result = super().forward(x, return_attention=return_attention)
        
        if return_attention:
            outputs, attention_dict = result
            # Add CBAM spatial attention to the dict
            attention_dict['cbam_spatial_attention'] = self.cbam_spatial_attention
            return outputs, attention_dict
        
        return result


def visualize_attention_comparison(model, video_tensor, frame_idx=None, save_dir='attention_comparison'):
    """
    Visualize CBAM spatial attention vs SpatialExplainabilityModule attention side-by-side
    
    Args:
        model: MultiClassExplainableResNet3DWithCBAMAttention model
        video_tensor: Input video (B, C, T, H, W)
        frame_idx: Specific frame index to visualize, or None to use most important frame
        save_dir: Directory to save visualizations
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    model.eval()
    device = next(model.parameters()).device
    video_tensor = video_tensor.to(device)
    
    with torch.no_grad():
        # Get attention maps
        outputs, attention_dict = model(video_tensor, return_attention=True)
        
        frame_importance = attention_dict['frame_importance']  # (B, T)
        spatial_attention = attention_dict['spatial_attention']  # (B, 1, T, H, W)
        cbam_spatial_attention = attention_dict['cbam_spatial_attention']  # (B, 1, T, H, W)
        
        B, C, T, H, W = video_tensor.shape
        
        # Process each video in batch
        for b in range(B):
            # Determine which frame to visualize
            if frame_idx is None:
                # Use the most important frame
                most_important_frame = torch.argmax(frame_importance[b]).item()
            else:
                most_important_frame = min(frame_idx, T - 1)
            
            # Get the original frame
            original_frame = video_tensor[b, :, most_important_frame, :, :].cpu()  # (C, H, W)
            original_frame = original_frame.permute(1, 2, 0).numpy()  # (H, W, C)
            
            # Normalize to [0, 1] for visualization
            original_frame = (original_frame - original_frame.min()) / (original_frame.max() - original_frame.min() + 1e-8)
            
            # Get attention maps for this frame
            cbam_attn = cbam_spatial_attention[b, 0, most_important_frame, :, :].cpu().numpy()  # (H, W)
            spatial_attn = spatial_attention[b, 0, most_important_frame, :, :].cpu().numpy()  # (H, W)
            
            # Upsample attention maps to original resolution if needed
            if cbam_attn.shape != (H, W):
                cbam_attn = cv2.resize(cbam_attn, (W, H), interpolation=cv2.INTER_LINEAR)
            if spatial_attn.shape != (H, W):
                spatial_attn = cv2.resize(spatial_attn, (W, H), interpolation=cv2.INTER_LINEAR)
            
            # Create visualization
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # Row 1: CBAM Spatial Attention
            axes[0, 0].imshow(original_frame)
            axes[0, 0].set_title('Original Frame', fontsize=12, fontweight='bold')
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(cbam_attn, cmap='jet', vmin=0, vmax=1)
            axes[0, 1].set_title('CBAM Spatial Attention', fontsize=12, fontweight='bold')
            axes[0, 1].axis('off')
            cbar1 = plt.colorbar(axes[0, 1].images[0], ax=axes[0, 1], fraction=0.046)
            cbar1.set_label('Attention Weight', fontsize=10)
            
            # Overlay CBAM attention on original frame
            overlay_cbam = original_frame.copy()
            heatmap_cbam = cv2.applyColorMap((cbam_attn * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heatmap_cbam = cv2.cvtColor(heatmap_cbam, cv2.COLOR_BGR2RGB) / 255.0
            overlay_cbam = 0.6 * overlay_cbam + 0.4 * heatmap_cbam
            axes[0, 2].imshow(overlay_cbam)
            axes[0, 2].set_title('CBAM Overlay', fontsize=12, fontweight='bold')
            axes[0, 2].axis('off')
            
            # Row 2: SpatialExplainabilityModule Attention
            axes[1, 0].imshow(original_frame)
            axes[1, 0].set_title('Original Frame', fontsize=12, fontweight='bold')
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(spatial_attn, cmap='jet', vmin=0, vmax=1)
            axes[1, 1].set_title('Spatial Explainability Attention', fontsize=12, fontweight='bold')
            axes[1, 1].axis('off')
            cbar2 = plt.colorbar(axes[1, 1].images[0], ax=axes[1, 1], fraction=0.046)
            cbar2.set_label('Attention Weight', fontsize=10)
            
            # Overlay spatial explainability attention on original frame
            overlay_spatial = original_frame.copy()
            heatmap_spatial = cv2.applyColorMap((spatial_attn * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heatmap_spatial = cv2.cvtColor(heatmap_spatial, cv2.COLOR_BGR2RGB) / 255.0
            overlay_spatial = 0.6 * overlay_spatial + 0.4 * heatmap_spatial
            axes[1, 2].imshow(overlay_spatial)
            axes[1, 2].set_title('Spatial Explainability Overlay', fontsize=12, fontweight='bold')
            axes[1, 2].axis('off')
            
            # Add statistics
            cbam_stats = f"Mean: {cbam_attn.mean():.4f}, Std: {cbam_attn.std():.4f}, Max: {cbam_attn.max():.4f}"
            spatial_stats = f"Mean: {spatial_attn.mean():.4f}, Std: {spatial_attn.std():.4f}, Max: {spatial_attn.max():.4f}"
            
            fig.suptitle(f'Attention Comparison - Video {b}, Frame {most_important_frame} (Importance: {frame_importance[b, most_important_frame]:.4f})\n'
                        f'CBAM Stats: {cbam_stats}\n'
                        f'Spatial Explainability Stats: {spatial_stats}',
                        fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(save_dir / f'attention_comparison_video{b}_frame{most_important_frame}.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Saved visualization for video {b}, frame {most_important_frame}")
            
            # Print sparsity analysis
            print(f"\n=== Sparsity Analysis for Video {b}, Frame {most_important_frame} ===")
            print(f"CBAM Spatial Attention:")
            print(f"  Mean: {cbam_attn.mean():.4f}")
            print(f"  Std: {cbam_attn.std():.4f}")
            print(f"  Max: {cbam_attn.max():.4f}")
            print(f"  Min: {cbam_attn.min():.4f}")
            print(f"  Pixels > 0.5: {(cbam_attn > 0.5).sum()} / {cbam_attn.size} ({(cbam_attn > 0.5).sum() / cbam_attn.size * 100:.2f}%)")
            print(f"  Pixels > 0.7: {(cbam_attn > 0.7).sum()} / {cbam_attn.size} ({(cbam_attn > 0.7).sum() / cbam_attn.size * 100:.2f}%)")
            
            print(f"\nSpatial Explainability Attention:")
            print(f"  Mean: {spatial_attn.mean():.4f}")
            print(f"  Std: {spatial_attn.std():.4f}")
            print(f"  Max: {spatial_attn.max():.4f}")
            print(f"  Min: {spatial_attn.min():.4f}")
            print(f"  Pixels > 0.5: {(spatial_attn > 0.5).sum()} / {spatial_attn.size} ({(spatial_attn > 0.5).sum() / spatial_attn.size * 100:.2f}%)")
            print(f"  Pixels > 0.7: {(spatial_attn > 0.7).sum()} / {spatial_attn.size} ({(spatial_attn > 0.7).sum() / spatial_attn.size * 100:.2f}%)")
            print("=" * 60)


def visualize_multiple_frames(model, video_tensor, num_frames=5, save_dir='attention_comparison_multi'):
    """
    Visualize attention for multiple important frames
    
    Args:
        model: MultiClassExplainableResNet3DWithCBAMAttention model
        video_tensor: Input video (B, C, T, H, W)
        num_frames: Number of top frames to visualize
        save_dir: Directory to save visualizations
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    model.eval()
    device = next(model.parameters()).device
    video_tensor = video_tensor.to(device)
    
    with torch.no_grad():
        # Get attention maps
        outputs, attention_dict = model(video_tensor, return_attention=True)
        
        frame_importance = attention_dict['frame_importance']  # (B, T)
        spatial_attention = attention_dict['spatial_attention']  # (B, 1, T, H, W)
        cbam_spatial_attention = attention_dict['cbam_spatial_attention']  # (B, 1, T, H, W)
        
        B, C, T, H, W = video_tensor.shape
        
        # Process first video in batch
        b = 0
        
        # Get top-k important frames
        num_frames = min(num_frames, T)
        top_scores, top_indices = torch.topk(frame_importance[b], num_frames)
        
        # Create a large figure with all frames
        fig, axes = plt.subplots(num_frames, 3, figsize=(15, 5 * num_frames))
        
        if num_frames == 1:
            axes = axes.reshape(1, -1)
        
        for idx, frame_idx in enumerate(top_indices):
            frame_idx = frame_idx.item()
            importance_score = top_scores[idx].item()
            
            # Get the original frame
            original_frame = video_tensor[b, :, frame_idx, :, :].cpu()
            original_frame = original_frame.permute(1, 2, 0).numpy()
            original_frame = (original_frame - original_frame.min()) / (original_frame.max() - original_frame.min() + 1e-8)
            
            # Get attention maps
            cbam_attn = cbam_spatial_attention[b, 0, frame_idx, :, :].cpu().numpy()
            spatial_attn = spatial_attention[b, 0, frame_idx, :, :].cpu().numpy()
            
            # Upsample if needed
            if cbam_attn.shape != (H, W):
                cbam_attn = cv2.resize(cbam_attn, (W, H), interpolation=cv2.INTER_LINEAR)
            if spatial_attn.shape != (H, W):
                spatial_attn = cv2.resize(spatial_attn, (W, H), interpolation=cv2.INTER_LINEAR)
            
            # Original frame
            axes[idx, 0].imshow(original_frame)
            axes[idx, 0].set_title(f'Frame {frame_idx} (Importance: {importance_score:.4f})', fontsize=11, fontweight='bold')
            axes[idx, 0].axis('off')
            
            # CBAM overlay
            overlay_cbam = original_frame.copy()
            heatmap_cbam = cv2.applyColorMap((cbam_attn * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heatmap_cbam = cv2.cvtColor(heatmap_cbam, cv2.COLOR_BGR2RGB) / 255.0
            overlay_cbam = 0.6 * overlay_cbam + 0.4 * heatmap_cbam
            axes[idx, 1].imshow(overlay_cbam)
            axes[idx, 1].set_title(f'CBAM (Sparsity: {(cbam_attn > 0.5).sum() / cbam_attn.size * 100:.1f}%)', fontsize=11)
            axes[idx, 1].axis('off')
            
            # Spatial Explainability overlay
            overlay_spatial = original_frame.copy()
            heatmap_spatial = cv2.applyColorMap((spatial_attn * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heatmap_spatial = cv2.cvtColor(heatmap_spatial, cv2.COLOR_BGR2RGB) / 255.0
            overlay_spatial = 0.6 * overlay_spatial + 0.4 * heatmap_spatial
            axes[idx, 2].imshow(overlay_spatial)
            axes[idx, 2].set_title(f'Spatial Explainability (Sparsity: {(spatial_attn > 0.5).sum() / spatial_attn.size * 100:.1f}%)', fontsize=11)
            axes[idx, 2].axis('off')
        
        fig.suptitle('Top Important Frames: CBAM vs Spatial Explainability Attention', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_dir / f'attention_comparison_top{num_frames}_frames.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved multi-frame visualization to {save_dir}")


if __name__ == "__main__":
    # Example usage
    print("Loading model...")
    model = MultiClassExplainableResNet3DWithCBAMAttention(
        num_diagnostic_classes=2,
        num_subtype_classes=4,
        pretrained=False,
        use_attention=True
    )
    
    # Load trained weights if available
    checkpoint_path = Path("checkpoints/best_model.pth")
    if checkpoint_path.exists():
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        print("No checkpoint found, using random weights for demonstration")
    
    # Create dummy input
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Generate random video or load real data
    dummy_video = torch.randn(1, 3, 32, 224, 224)  # (B=1, C=3, T=32, H=224, W=224)
    
    print("\nVisualizing single frame comparison...")
    visualize_attention_comparison(model, dummy_video, frame_idx=None, save_dir='attention_comparison')
    
    print("\nVisualizing multiple important frames...")
    visualize_multiple_frames(model, dummy_video, num_frames=5, save_dir='attention_comparison_multi')
    
    print("\nVisualization complete!")
