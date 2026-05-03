#!/usr/bin/env python3
"""
Inference script for ERDES Medical Video Diagnosis

This script performs inference on a single video using the trained pipeline:
1. Loads the trained multi-class classifier
2. Extracts important frames and attention maps
3. Loads the finetuned VLM (if available)
4. Generates clinical diagnosis with reasoning

Usage:
    # Classifier-only inference (predictions + attention visualization)
    python run_inference.py --video_path /path/to/video.mp4 \
        --classifier_checkpoint ./checkpoints/multiclass/best_model_weights.pth

    # Full pipeline with VLM (predictions + clinical reasoning)
    python run_inference.py --video_path /path/to/video.mp4 \
        --classifier_checkpoint ./checkpoints/multiclass/best_model_weights.pth \
        --vlm_checkpoint ./checkpoints/vlm_finetuned/vlm_checkpoints/final_model

    # With custom output directory
    python run_inference.py --video_path /path/to/video.mp4 \
        --classifier_checkpoint ./checkpoints/multiclass/best_model_weights.pth \
        --output_dir ./inference_results
"""

import torch
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import sys

from multiclass_model import create_multiclass_model
from vlm_data_preparation import VLMDataPreparator
from vlm_finetuning import setup_qwen2vl_for_finetuning, inference_vlm


def load_video(video_path: str, num_frames: int = 32, img_size: int = 224):
    """
    Load and preprocess video
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to sample
        img_size: Target image size
        
    Returns:
        video_tensor: Preprocessed video tensor (1, C, T, H, W)
    """
    print(f"Loading video: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"  Total frames: {total_frames}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Duration: {total_frames/fps:.2f}s")
    
    # Sample frames uniformly
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Could not read frame {idx}")
            continue
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize
        frame = cv2.resize(frame, (img_size, img_size))
        
        # Normalize to [0, 1]
        frame = frame.astype(np.float32) / 255.0
        
        frames.append(frame)
    
    cap.release()
    
    if len(frames) != num_frames:
        raise ValueError(f"Expected {num_frames} frames, got {len(frames)}")
    
    # Convert to tensor: (T, H, W, C) -> (C, T, H, W)
    video = np.stack(frames, axis=0)  # (T, H, W, C)
    video = np.transpose(video, (3, 0, 1, 2))  # (C, T, H, W)
    video_tensor = torch.from_numpy(video).unsqueeze(0)  # (1, C, T, H, W)
    
    print(f"  Loaded video tensor: {video_tensor.shape}")
    
    return video_tensor


def run_classifier_inference(model, video_tensor, device='cuda'):
    """
    Run classifier inference
    
    Args:
        model: Trained classifier model
        video_tensor: Video tensor (1, C, T, H, W)
        device: Device to use
        
    Returns:
        predictions: Dictionary with predictions and confidences
        attention: Dictionary with attention maps
    """
    print("\n" + "="*80)
    print("Running Classifier Inference")
    print("="*80)
    
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
    
    # Class labels
    diagnostic_labels = {0: "Non-RD", 1: "RD"}
    subtype_labels = {
        0: "Normal",
        1: "PVD",
        2: "Macula Intact",
        3: "Macula Detached"
    }
    
    predictions = {
        'diagnostic': diagnostic_labels.get(diagnostic_pred, f"Class {diagnostic_pred}"),
        'diagnostic_class': diagnostic_pred,
        'diagnostic_confidence': diagnostic_conf,
        'diagnostic_probs': diagnostic_probs[0].cpu().tolist(),
        'subtype': subtype_labels.get(subtype_pred, f"Class {subtype_pred}"),
        'subtype_class': subtype_pred,
        'subtype_confidence': subtype_conf,
        'subtype_probs': subtype_probs[0].cpu().tolist()
    }
    
    print("\nClassifier Predictions:")
    print(f"  Diagnostic: {predictions['diagnostic']} ({predictions['diagnostic_confidence']:.1%})")
    print(f"  Subtype: {predictions['subtype']} ({predictions['subtype_confidence']:.1%})")
    
    return predictions, attention


def save_results(video_path, predictions, attention, output_dir, vlm_reasoning=None):
    """
    Save inference results
    
    Args:
        video_path: Path to input video
        predictions: Classifier predictions
        attention: Attention maps
        output_dir: Output directory
        vlm_reasoning: VLM clinical reasoning (optional)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    video_name = Path(video_path).stem
    
    # Save JSON results
    results = {
        'video_path': str(video_path),
        'video_name': video_name,
        'predictions': predictions,
        'frame_importance': attention['frame_importance'][0].cpu().tolist(),
    }
    
    if vlm_reasoning:
        results['clinical_reasoning'] = vlm_reasoning
    
    results_path = output_dir / f"{video_name}_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    
    # Save frame importance visualization
    import matplotlib.pyplot as plt
    
    frame_importance = attention['frame_importance'][0].cpu().numpy()
    
    plt.figure(figsize=(12, 4))
    plt.bar(range(len(frame_importance)), frame_importance)
    plt.xlabel('Frame Index')
    plt.ylabel('Importance Score')
    plt.title('Frame Importance Scores')
    plt.grid(True, alpha=0.3)
    
    importance_plot_path = output_dir / f"{video_name}_frame_importance.png"
    plt.savefig(importance_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Frame importance plot saved to: {importance_plot_path}")
    
    return results_path


def main():
    parser = argparse.ArgumentParser(
        description='Run inference on a video using trained ERDES models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Required arguments
    parser.add_argument('--video_path', type=str, required=True,
                       help='Path to input video file')
    parser.add_argument('--classifier_checkpoint', type=str, required=True,
                       help='Path to trained classifier checkpoint')
    
    # Optional arguments
    parser.add_argument('--vlm_checkpoint', type=str, default=None,
                       help='Path to finetuned VLM checkpoint (optional)')
    parser.add_argument('--output_dir', type=str, default='./inference_output',
                       help='Output directory for results')
    
    # Model configuration
    parser.add_argument('--num_diagnostic_classes', type=int, default=2,
                       help='Number of diagnostic classes')
    parser.add_argument('--num_subtype_classes', type=int, default=4,
                       help='Number of subtype classes')
    parser.add_argument('--num_frames', type=int, default=32,
                       help='Number of frames to sample from video')
    parser.add_argument('--img_size', type=int, default=224,
                       help='Image size for preprocessing')
    parser.add_argument('--top_k_frames', type=int, default=5,
                       help='Number of important frames to extract for VLM')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use for inference')
    
    args = parser.parse_args()
    
    # Check if video exists
    if not Path(args.video_path).exists():
        print(f"Error: Video file not found: {args.video_path}")
        sys.exit(1)
    
    # Check if classifier checkpoint exists
    if not Path(args.classifier_checkpoint).exists():
        print(f"Error: Classifier checkpoint not found: {args.classifier_checkpoint}")
        sys.exit(1)
    
    # Set device
    device = args.device if torch.cuda.is_available() else 'cpu'
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
    
    print("="*80)
    print("ERDES Medical Video Diagnosis - Inference")
    print("="*80)
    print(f"Video: {args.video_path}")
    print(f"Classifier: {args.classifier_checkpoint}")
    print(f"VLM: {args.vlm_checkpoint if args.vlm_checkpoint else 'Not used'}")
    print(f"Device: {device}")
    print("="*80)
    
    # Step 1: Load video
    video_tensor = load_video(
        args.video_path,
        num_frames=args.num_frames,
        img_size=args.img_size
    )
    
    # Step 2: Load classifier
    print("\nLoading classifier model...")
    model = create_multiclass_model(
        num_diagnostic_classes=args.num_diagnostic_classes,
        num_subtype_classes=args.num_subtype_classes,
        pretrained=False,
        dropout=0.3
    )
    
    checkpoint = torch.load(args.classifier_checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    print("Classifier loaded successfully!")
    
    # Step 3: Run classifier inference
    predictions, attention = run_classifier_inference(model, video_tensor, device)
    
    # Step 4: VLM inference (if checkpoint provided)
    vlm_reasoning = None
    if args.vlm_checkpoint:
        print("\n" + "="*80)
        print("Running VLM Inference")
        print("="*80)
        
        # Check if VLM checkpoint exists
        if not Path(args.vlm_checkpoint).exists():
            print(f"Warning: VLM checkpoint not found: {args.vlm_checkpoint}")
            print("Skipping VLM inference")
        else:
            try:
                # Prepare VLM data
                print("\nPreparing data for VLM...")
                preparator = VLMDataPreparator(
                    model=model,
                    device=device,
                    top_k_frames=args.top_k_frames
                )
                
                temp_dir = Path(args.output_dir) / "temp_vlm_data"
                sample = preparator.prepare_vlm_sample(
                    video_tensor=video_tensor,
                    video_id=Path(args.video_path).stem,
                    output_dir=str(temp_dir)
                )
                
                print(f"  Extracted {len(sample['frame_indices'])} important frames")
                print(f"  Frame indices: {sample['frame_indices']}")
                
                # Load VLM
                print("\nLoading VLM model...")
                vlm_model, vlm_processor = setup_qwen2vl_for_finetuning(
                    model_name=args.vlm_checkpoint,
                    use_lora=False,
                    load_in_4bit=False
                )
                print("VLM loaded successfully!")
                
                # Run VLM inference
                print("\nGenerating clinical reasoning...")
                vlm_reasoning = inference_vlm(
                    model=vlm_model,
                    processor=vlm_processor,
                    image_paths=sample['heatmap_paths'],
                    prompt=sample['prompt'],
                    device=device
                )
                
                print("\nClinical Reasoning:")
                print("-" * 80)
                print(vlm_reasoning)
                print("-" * 80)
                
            except Exception as e:
                print(f"Error during VLM inference: {e}")
                print("Continuing with classifier results only")
    
    # Step 5: Save results
    print("\n" + "="*80)
    print("Saving Results")
    print("="*80)
    
    results_path = save_results(
        args.video_path,
        predictions,
        attention,
        args.output_dir,
        vlm_reasoning
    )
    
    print("\n" + "="*80)
    print("Inference Complete!")
    print("="*80)
    print(f"\nResults saved to: {args.output_dir}")
    print(f"  - JSON results: {results_path}")
    print(f"  - Frame importance plot: {Path(args.output_dir) / f'{Path(args.video_path).stem}_frame_importance.png'}")
    
    if vlm_reasoning:
        print(f"\n✓ Full pipeline completed (Classifier + VLM)")
    else:
        print(f"\n✓ Classifier inference completed")
        print(f"  To run full pipeline with clinical reasoning, provide --vlm_checkpoint")


if __name__ == "__main__":
    main()
