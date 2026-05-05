#!/usr/bin/env python3
"""
Generate Figure 2: Attention Visualization Examples

This script creates a multi-panel figure showing:
- Original video frames
- Frame importance scores (bar chart)
- Spatial attention maps overlaid on key frames
- Ground truth labels and predictions

Usage:
    python generate_figure2_attention.py \
        --model_checkpoint ../model/checkpoints/multiclass/best_model_weights.pth \
        --data_csv ../benchmarks/input/balanced_split_desc.csv \
        --output_dir ./figures \
        --num_examples 6
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import seaborn as sns
import cv2
import torch
from pathlib import Path
from typing import List, Dict, Tuple

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'model'))
from multiclass_model import create_multiclass_model


def load_video(video_path: str, num_frames: int = 32, img_size: int = 224):
    """Load and preprocess video"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return None, None
    
    # Sample frames uniformly
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    original_frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Store original for visualization
        original_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Preprocess for model
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (img_size, img_size))
        frame = frame.astype(np.float32) / 255.0
        frames.append(frame)
    
    cap.release()
    
    if len(frames) != num_frames:
        return None, None
    
    # Convert to tensor
    video = np.stack(frames, axis=0)
    video = np.transpose(video, (3, 0, 1, 2))
    video_tensor = torch.from_numpy(video).unsqueeze(0)
    
    # Apply ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)
    video_tensor = (video_tensor - mean) / std
    
    return video_tensor, original_frames


def run_inference_with_attention(model, video_tensor, device):
    """Run inference and extract attention maps"""
    model.eval()
    video_tensor = video_tensor.to(device)
    
    with torch.no_grad():
        outputs, attention = model(video_tensor, return_attention=True)
    
    # Get predictions
    diagnostic_probs = torch.softmax(outputs['diagnostic'], dim=1)
    subtype_probs = torch.softmax(outputs['subtype'], dim=1)
    
    diagnostic_pred = torch.argmax(diagnostic_probs, dim=1).item()
    subtype_pred = torch.argmax(subtype_probs, dim=1).item()
    
    diagnostic_conf = diagnostic_probs[0, diagnostic_pred].item()
    subtype_conf = subtype_probs[0, subtype_pred].item()
    
    # Extract attention
    frame_importance = attention['frame_importance'][0].cpu().numpy()
    spatial_attention_raw = attention['spatial_attention'][0].cpu().numpy()
    
    # Debug: Print shapes
    print(f"  Frame importance shape: {frame_importance.shape}")
    print(f"  Spatial attention raw shape: {spatial_attention_raw.shape}")
    
    # Process spatial attention
    # Shape is typically [1, T, H, W] or [T, H, W]
    # We want to aggregate to [H, W]
    if len(spatial_attention_raw.shape) == 4:
        # [1, T, H, W] -> [T, H, W]
        spatial_attention_raw = spatial_attention_raw[0]
    
    if len(spatial_attention_raw.shape) == 3:
        # [T, H, W] -> [H, W] by averaging across temporal dimension
        spatial_attention = np.mean(spatial_attention_raw, axis=0)
        print(f"  Spatial attention aggregated shape: {spatial_attention.shape}")
    else:
        # Already [H, W]
        spatial_attention = spatial_attention_raw
    
    return {
        'diagnostic_pred': diagnostic_pred,
        'diagnostic_conf': diagnostic_conf,
        'subtype_pred': subtype_pred,
        'subtype_conf': subtype_conf,
        'frame_importance': frame_importance,
        'spatial_attention': spatial_attention
    }


def overlay_heatmap(image, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """Overlay spatial attention heatmap on image"""
    # Get image dimensions
    h, w = image.shape[:2]
    
    # Handle edge cases
    if heatmap is None or heatmap.size == 0:
        print(f"Warning: Empty heatmap, returning original image")
        return image, np.zeros_like(image)
    
    # Ensure heatmap is 2D
    if len(heatmap.shape) == 3:
        # If heatmap has channels, take first channel or average
        if heatmap.shape[0] == 1:
            heatmap = heatmap[0]
        elif heatmap.shape[2] == 1:
            heatmap = heatmap[:, :, 0]
        else:
            heatmap = np.mean(heatmap, axis=-1)
    
    # Check heatmap shape
    if len(heatmap.shape) != 2:
        print(f"Warning: Invalid heatmap shape {heatmap.shape}, returning original image")
        return image, np.zeros_like(image)
    
    # Resize heatmap to match image
    try:
        heatmap_resized = cv2.resize(heatmap, (w, h))
    except cv2.error as e:
        print(f"Warning: Failed to resize heatmap from {heatmap.shape} to ({w}, {h}): {e}")
        return image, np.zeros_like(image)
    
    # Normalize heatmap to 0-255
    heatmap_min = heatmap_resized.min()
    heatmap_max = heatmap_resized.max()
    
    if heatmap_max - heatmap_min < 1e-8:
        # Uniform heatmap, return original image
        print(f"Warning: Uniform heatmap (min={heatmap_min:.4f}, max={heatmap_max:.4f}), returning original image")
        return image, np.zeros_like(image)
    
    heatmap_normalized = ((heatmap_resized - heatmap_min) / 
                          (heatmap_max - heatmap_min) * 255).astype(np.uint8)
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap_normalized, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Blend with original image
    overlaid = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    
    return overlaid, heatmap_colored


def create_attention_visualization(
    original_frames: List[np.ndarray],
    results: Dict,
    ground_truth: Dict,
    video_name: str,
    save_path: Path
):
    """
    Create comprehensive attention visualization for one video
    
    Layout:
    - Row 1: 5 key frames (original)
    - Row 2: 5 key frames with spatial attention overlays
    - Row 3: Frame importance bar chart
    - Row 4: Prediction vs ground truth text
    """
    # Set up the figure
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(4, 5, figure=fig, height_ratios=[2, 2, 1.5, 0.5], hspace=0.3, wspace=0.2)
    
    # Get top 5 frames by importance
    frame_importance = results['frame_importance']
    top_k_indices = np.argsort(frame_importance)[-5:][::-1]
    
    # Label mappings
    diagnostic_labels = {0: "Non-RD", 1: "RD"}
    subtype_labels = {0: "Normal", 1: "Macula Intact", 2: "Macula Detached", 3: "PVD"}
    
    # Get spatial attention (single aggregated map)
    spatial_attn = results['spatial_attention']
    
    # Row 1: Original frames
    for i, frame_idx in enumerate(top_k_indices):
        ax = fig.add_subplot(gs[0, i])
        
        frame = original_frames[frame_idx]
        ax.imshow(frame)
        ax.set_title(f'Frame {frame_idx} (Original)\nImportance: {frame_importance[frame_idx]:.3f}', 
                     fontsize=10, fontweight='bold')
        ax.axis('off')
    
    # Row 2: Attention overlays
    for i, frame_idx in enumerate(top_k_indices):
        ax = fig.add_subplot(gs[1, i])
        
        frame = original_frames[frame_idx]
        overlaid, _ = overlay_heatmap(frame, spatial_attn, alpha=0.5)
        
        ax.imshow(overlaid)
        ax.set_title(f'Frame {frame_idx} (Attention)', 
                     fontsize=10, fontweight='bold')
        ax.axis('off')
    
    # Row 3: Frame importance bar chart (spans all columns)
    ax_bar = fig.add_subplot(gs[2, :])
    
    frames_range = np.arange(len(frame_importance))
    colors = ['#d62728' if i in top_k_indices else '#1f77b4' for i in frames_range]
    
    ax_bar.bar(frames_range, frame_importance, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax_bar.set_xlabel('Frame Index', fontsize=12, fontweight='bold')
    ax_bar.set_ylabel('Importance Score', fontsize=12, fontweight='bold')
    ax_bar.set_title('Frame Importance Scores (Top 5 highlighted in red)', fontsize=12, fontweight='bold')
    ax_bar.grid(axis='y', alpha=0.3)
    ax_bar.set_xlim(-1, len(frame_importance))
    
    # Row 4: Prediction and ground truth info (spans all columns)
    ax_text = fig.add_subplot(gs[3, :])
    ax_text.axis('off')
    
    # Prepare text
    pred_diag = diagnostic_labels[results['diagnostic_pred']]
    pred_sub = subtype_labels[results['subtype_pred']]
    gt_diag = diagnostic_labels.get(ground_truth.get('diagnostic', -1), 'Unknown')
    gt_sub = subtype_labels.get(ground_truth.get('subtype', -1), 'Unknown')
    
    # Check if prediction is correct
    diag_correct = results['diagnostic_pred'] == ground_truth.get('diagnostic', -1)
    sub_correct = results['subtype_pred'] == ground_truth.get('subtype', -1)
    
    diag_color = 'green' if diag_correct else 'red'
    sub_color = 'green' if sub_correct else 'red'
    
    info_text = (
        f"Video: {video_name}\n"
        f"Prediction: {pred_diag} ({results['diagnostic_conf']:.2%}) | {pred_sub} ({results['subtype_conf']:.2%})\n"
        f"Ground Truth: {gt_diag} | {gt_sub}"
    )
    
    ax_text.text(0.5, 0.5, info_text, 
                ha='center', va='center', fontsize=11, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
                family='monospace')
    
    # Add colored boxes for correctness
    if diag_correct and sub_correct:
        fig.patch.set_facecolor('#e8f5e9')  # Light green background
    elif not diag_correct:
        fig.patch.set_facecolor('#ffebee')  # Light red background
    
    plt.suptitle(f'Attention Visualization: {video_name}', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    
    print(f"Saved visualization to: {save_path}")


def create_combined_figure(
    examples: List[Dict],
    output_path: Path
):
    """
    Create Figure 2: Combined multi-example visualization
    
    Shows 4-6 examples in a grid format with compact layout
    Each row shows: original frames (top) + attention overlays (bottom)
    """
    num_examples = len(examples)
    
    # Create figure with subplots - now with 2 rows per example (original + attention)
    fig = plt.figure(figsize=(26, 3 * num_examples))
    
    # Label mappings
    diagnostic_labels = {0: "Non-RD", 1: "RD"}
    subtype_labels = {0: "Normal", 1: "Macula Intact", 2: "Macula Detached", 3: "PVD"}
    
    for ex_idx, example in enumerate(examples):
        # Create grid for this example - 2 rows: original frames + attention overlays
        gs = gridspec.GridSpec(
            num_examples * 2, 7, 
            figure=fig,
            height_ratios=[0.8, 0.8] * num_examples,
            hspace=0.15, wspace=0.2,
            top=0.97 - ex_idx * (0.97 / num_examples),
            bottom=0.97 - (ex_idx + 1) * (0.97 / num_examples)
        )
        
        results = example['results']
        original_frames = example['frames']
        ground_truth = example['ground_truth']
        video_name = example['video_name']
        
        # Get top 5 frames
        frame_importance = results['frame_importance']
        top_k_indices = np.argsort(frame_importance)[-5:][::-1]
        
        # Get spatial attention (single aggregated map)
        spatial_attn = results['spatial_attention']
        
        # Row 1: Original frames
        for i, frame_idx in enumerate(top_k_indices):
            ax = fig.add_subplot(gs[ex_idx * 2, i])
            
            frame = original_frames[frame_idx]
            ax.imshow(frame)
            
            if ex_idx == 0:
                ax.set_title(f'Top {i+1} (Original)', fontsize=9, fontweight='bold')
            ax.axis('off')
        
        # Row 2: Attention overlays
        for i, frame_idx in enumerate(top_k_indices):
            ax = fig.add_subplot(gs[ex_idx * 2 + 1, i])
            
            frame = original_frames[frame_idx]
            overlaid, _ = overlay_heatmap(frame, spatial_attn, alpha=0.5)
            
            ax.imshow(overlaid)
            if ex_idx == 0:
                ax.set_title(f'Top {i+1} (Attention)', fontsize=9, fontweight='bold')
            ax.axis('off')
        
        # Frame importance chart (spans both rows)
        ax_bar = fig.add_subplot(gs[ex_idx * 2:ex_idx * 2 + 2, 5])
        frames_range = np.arange(len(frame_importance))
        colors = ['#d62728' if i in top_k_indices else '#1f77b4' for i in frames_range]
        ax_bar.bar(frames_range, frame_importance, color=colors, alpha=0.7, width=1.0)
        ax_bar.set_ylim(0, frame_importance.max() * 1.1)
        if ex_idx == 0:
            ax_bar.set_title('Frame Importance', fontsize=10, fontweight='bold')
        ax_bar.set_xticks([])
        ax_bar.set_yticks([])
        ax_bar.spines['top'].set_visible(False)
        ax_bar.spines['right'].set_visible(False)
        
        # Info panel (spans both rows)
        ax_info = fig.add_subplot(gs[ex_idx * 2:ex_idx * 2 + 2, 6])
        ax_info.axis('off')
        
        pred_diag = diagnostic_labels[results['diagnostic_pred']]
        pred_sub = subtype_labels[results['subtype_pred']]
        gt_diag = diagnostic_labels.get(ground_truth.get('diagnostic', -1), 'Unknown')
        gt_sub = subtype_labels.get(ground_truth.get('subtype', -1), 'Unknown')
        
        diag_correct = results['diagnostic_pred'] == ground_truth.get('diagnostic', -1)
        sub_correct = results['subtype_pred'] == ground_truth.get('subtype', -1)
        
        status = "✓ Correct" if (diag_correct and sub_correct) else "✗ Incorrect"
        status_color = 'green' if (diag_correct and sub_correct) else 'red'
        
        info_text = (
            f"{video_name}\n\n"
            f"Pred: {pred_diag}\n"
            f"      {pred_sub}\n\n"
            f"GT:   {gt_diag}\n"
            f"      {gt_sub}\n\n"
            f"{status}"
        )
        
        ax_info.text(0.1, 0.5, info_text, 
                    ha='left', va='center', fontsize=9,
                    family='monospace',
                    bbox=dict(boxstyle='round', 
                             facecolor='lightgreen' if diag_correct and sub_correct else 'lightcoral',
                             alpha=0.3))
    
    plt.suptitle('Figure 2: Attention Visualization Examples', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved combined Figure 2 to: {output_path}")


def select_diverse_examples(
    df: pd.DataFrame,
    num_examples: int = 6
) -> List[int]:
    """
    Select diverse examples covering different classes and scenarios
    
    Strategy:
    - 2 correct RD predictions (1 macula intact, 1 macula detached)
    - 2 correct non-RD predictions (1 normal, 1 PVD)
    - 1-2 challenging/incorrect cases
    """
    selected_indices = []
    
    # Label mappings
    diagnostic_map = {'non_rd': 0, 'rd': 1}
    subtype_map = {'normal': 0, 'macula_intact': 1, 'macula_detached': 2, 'pvd': 3}
    
    # Map labels
    df['diagnostic_class_num'] = df['diagnostic_class'].map(diagnostic_map)
    df['subtype_num'] = df['subtype'].map(subtype_map)
    
    # Select examples
    # 1. RD + Macula Detached
    rd_mac_det = df[(df['diagnostic_class_num'] == 1) & (df['subtype_num'] == 2)]
    if len(rd_mac_det) > 0:
        selected_indices.append(rd_mac_det.sample(1).index[0])
    
    # 2. RD + Macula Intact
    rd_mac_int = df[(df['diagnostic_class_num'] == 1) & (df['subtype_num'] == 1)]
    if len(rd_mac_int) > 0:
        selected_indices.append(rd_mac_int.sample(1).index[0])
    
    # 3. Non-RD + Normal
    non_rd_normal = df[(df['diagnostic_class_num'] == 0) & (df['subtype_num'] == 0)]
    if len(non_rd_normal) > 0:
        selected_indices.append(non_rd_normal.sample(1).index[0])
    
    # 4. Non-RD + PVD
    non_rd_pvd = df[(df['diagnostic_class_num'] == 0) & (df['subtype_num'] == 3)]
    if len(non_rd_pvd) > 0:
        selected_indices.append(non_rd_pvd.sample(1).index[0])
    
    # 5-6. Fill remaining with random samples
    remaining = num_examples - len(selected_indices)
    if remaining > 0:
        available = df.index.difference(selected_indices)
        if len(available) >= remaining:
            selected_indices.extend(available.to_series().sample(remaining).tolist())
    
    return selected_indices[:num_examples]


def main():
    parser = argparse.ArgumentParser(
        description='Generate Figure 2: Attention Visualization Examples'
    )
    
    parser.add_argument('--model_checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--data_csv', type=str, required=True,
                       help='Path to dataset CSV')
    parser.add_argument('--video_base_dir', type=str, default=None,
                       help='Base directory for video files')
    parser.add_argument('--output_dir', type=str, default='./figures',
                       help='Output directory for figures')
    parser.add_argument('--num_examples', type=int, default=6,
                       help='Number of examples to visualize')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--num_frames', type=int, default=32,
                       help='Number of frames to sample')
    parser.add_argument('--img_size', type=int, default=224,
                       help='Image size for preprocessing')
    
    args = parser.parse_args()
    
    # Setup
    device = args.device if torch.cuda.is_available() else 'cpu'
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("Generating Figure 2: Attention Visualization Examples")
    print("="*80)
    print(f"Model: {args.model_checkpoint}")
    print(f"Data: {args.data_csv}")
    print(f"Output: {output_dir}")
    print(f"Device: {device}")
    print(f"Examples: {args.num_examples}")
    print("="*80 + "\n")
    
    # Load model
    print("Loading model...")
    model = create_multiclass_model(
        num_diagnostic_classes=2,
        num_subtype_classes=4,
        pretrained=False
    )
    
    checkpoint = torch.load(args.model_checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    print("Model loaded successfully\n")
    
    # Load data
    print("Loading dataset...")
    df = pd.read_csv(args.data_csv)
    
    # Select diverse examples
    selected_indices = select_diverse_examples(df, args.num_examples)
    print(f"Selected {len(selected_indices)} diverse examples\n")
    
    # Process each example
    examples = []
    
    for idx in selected_indices:
        row = df.iloc[idx]
        
        # Get video path
        if 'file_path' in row:
            video_path = row['file_path']
        elif 'video_path' in row:
            video_path = row['video_path']
        else:
            print(f"Warning: No video path for index {idx}")
            continue
        
        # Make path absolute
        if not Path(video_path).is_absolute():
            if args.video_base_dir:
                video_path = str(Path(args.video_base_dir) / video_path)
            else:
                base_dir = Path(args.data_csv).parent
                video_path = str(base_dir / video_path)
        
        video_name = Path(video_path).stem
        
        print(f"Processing: {video_name}")
        
        # Load video
        video_tensor, original_frames = load_video(
            video_path, 
            num_frames=args.num_frames,
            img_size=args.img_size
        )
        
        if video_tensor is None:
            print(f"  Failed to load video: {video_path}")
            continue
        
        # Run inference
        results = run_inference_with_attention(model, video_tensor, device)
        
        # Get ground truth
        diagnostic_map = {'non_rd': 0, 'rd': 1}
        subtype_map = {'normal': 0, 'macula_intact': 1, 'macula_detached': 2, 'pvd': 3}
        
        ground_truth = {
            'diagnostic': diagnostic_map.get(row.get('diagnostic_class', ''), -1),
            'subtype': subtype_map.get(row.get('subtype', ''), -1)
        }
        
        # Create individual visualization
        individual_path = output_dir / f'attention_{video_name}.png'
        create_attention_visualization(
            original_frames, results, ground_truth, video_name, individual_path
        )
        
        # Store for combined figure
        examples.append({
            'video_name': video_name,
            'frames': original_frames,
            'results': results,
            'ground_truth': ground_truth
        })
        
        print(f"  ✓ Processed successfully\n")
    
    # Create combined Figure 2
    if len(examples) > 0:
        combined_path = output_dir / 'Figure2_Attention_Visualization.png'
        create_combined_figure(examples, combined_path)
    
    print("\n" + "="*80)
    print("Figure generation complete!")
    print(f"Individual visualizations: {output_dir}/attention_*.png")
    print(f"Combined Figure 2: {output_dir}/Figure2_Attention_Visualization.png")
    print("="*80)


if __name__ == '__main__':
    main()
