#!/usr/bin/env python3
"""
Batch Inference Script for ERDES Test Set

This script performs inference on the entire test set:
1. Loads the test split from balanced_split_desc.csv
2. Runs classifier inference on all test videos
3. Optionally runs VLM for clinical reasoning
4. Generates comprehensive evaluation metrics and visualizations

Usage:
    # Classifier-only inference
    python run_test_set_inference.py \
        --data_csv ./balanced_split_desc.csv \
        --classifier_checkpoint ./checkpoints/multiclass/best_model_weights.pth \
        --output_dir ./test_set_results

    # Full pipeline with VLM
    python run_test_set_inference.py \
        --data_csv ./balanced_split_desc.csv \
        --classifier_checkpoint ./checkpoints/multiclass/best_model_weights.pth \
        --vlm_checkpoint ./checkpoints/vlm_finetuned/vlm_checkpoints/final_model \
        --output_dir ./test_set_results

    # With parallel processing
    python run_test_set_inference.py \
        --data_csv ./balanced_split_desc.csv \
        --classifier_checkpoint ./checkpoints/multiclass/best_model_weights.pth \
        --num_workers 4 \
        --batch_size 8
"""

import torch
import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import cv2
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

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
        success: Whether video was loaded successfully
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, False
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            return None, False
        
        # Sample frames uniformly
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
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
            return None, False
        
        # Convert to tensor: (T, H, W, C) -> (C, T, H, W)
        video = np.stack(frames, axis=0)  # (T, H, W, C)
        video = np.transpose(video, (3, 0, 1, 2))  # (C, T, H, W)
        video_tensor = torch.from_numpy(video).unsqueeze(0)  # (1, C, T, H, W)
        
        return video_tensor, True
        
    except Exception as e:
        print(f"Error loading video {video_path}: {e}")
        return None, False


def run_batch_inference(
    model,
    test_df: pd.DataFrame,
    device: str = 'cuda',
    num_frames: int = 32,
    img_size: int = 224,
    save_attention: bool = True,
    output_dir: Path = None,
    video_base_dir: str = None
) -> Tuple[List[Dict], List[Dict]]:
    """
    Run inference on all test videos
    
    Args:
        model: Trained classifier model
        test_df: DataFrame with test set information
        device: Device to use
        num_frames: Number of frames to sample
        img_size: Image size
        save_attention: Whether to save attention maps
        output_dir: Output directory for attention maps
        video_base_dir: Base directory for video files (if paths are relative)
        
    Returns:
        predictions: List of prediction dictionaries
        attention_maps: List of attention map dictionaries
    """
    model.eval()
    
    predictions = []
    attention_maps = []
    
    # Class labels
    diagnostic_labels = {0: "Non-RD", 1: "RD"}
    subtype_labels = {
        0: "Normal",
        1: "PVD",
        2: "Macula Intact",
        3: "Macula Detached"
    }
    
    print(f"\nProcessing {len(test_df)} test videos...")
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Running inference"):
        # Handle both 'video_path' and 'file_path' column names
        if 'video_path' in row:
            video_path = row['video_path']
        elif 'file_path' in row:
            video_path = row['file_path']
        else:
            print(f"\nWarning: No video path column found in row {idx}")
            continue
        
        # If path is relative, make it absolute
        if not Path(video_path).is_absolute():
            if video_base_dir:
                # Use provided base directory
                video_path = str(Path(video_base_dir) / video_path)
            else:
                # Assume paths are relative to benchmarks/input/
                base_dir = Path(__file__).parent.parent / 'benchmarks' / 'input'
                video_path = str(base_dir / video_path)
        
        video_name = Path(video_path).stem
        
        # Load video
        video_tensor, success = load_video(video_path, num_frames, img_size)
        
        if not success:
            print(f"\nWarning: Failed to load video: {video_path}")
            # Add placeholder for failed videos
            predictions.append({
                'video_name': video_name,
                'video_path': video_path,
                'success': False,
                'error': 'Failed to load video'
            })
            continue
        
        # Run inference
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
        
        # Store predictions
        pred_dict = {
            'video_name': video_name,
            'video_path': video_path,
            'success': True,
            'diagnostic_pred': diagnostic_pred,
            'diagnostic_label': diagnostic_labels[diagnostic_pred],
            'diagnostic_confidence': diagnostic_conf,
            'diagnostic_probs': diagnostic_probs[0].cpu().tolist(),
            'subtype_pred': subtype_pred,
            'subtype_label': subtype_labels[subtype_pred],
            'subtype_confidence': subtype_conf,
            'subtype_probs': subtype_probs[0].cpu().tolist(),
            'diagnostic_true': row.get('diagnostic_class', -1),
            'subtype_true': row.get('subtype_class', -1),
        }
        
        predictions.append(pred_dict)
        
        # Store attention maps
        if save_attention:
            attn_dict = {
                'video_name': video_name,
                'frame_importance': attention['frame_importance'][0].cpu().numpy(),
                'spatial_attention': attention['spatial_attention'][0].cpu().numpy()
            }
            attention_maps.append(attn_dict)
            
            # Save attention maps to disk
            if output_dir:
                attn_dir = output_dir / 'attention_maps'
                attn_dir.mkdir(parents=True, exist_ok=True)
                np.savez(
                    attn_dir / f'{video_name}_attention.npz',
                    frame_importance=attn_dict['frame_importance'],
                    spatial_attention=attn_dict['spatial_attention']
                )
    
    return predictions, attention_maps


def compute_metrics(predictions: List[Dict]) -> Dict:
    """
    Compute evaluation metrics from predictions
    
    Args:
        predictions: List of prediction dictionaries
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    try:
        # Filter successful predictions
        valid_preds = [p for p in predictions if p.get('success', False)]
        
        if len(valid_preds) == 0:
            return {'error': 'No valid predictions'}
        
        # Extract predictions and ground truth
        diagnostic_preds = [p['diagnostic_pred'] for p in valid_preds]
        diagnostic_true = [p['diagnostic_true'] for p in valid_preds]
        subtype_preds = [p['subtype_pred'] for p in valid_preds]
        subtype_true = [p['subtype_true'] for p in valid_preds]
        
        # Check for invalid ground truth labels
        if any(x == -1 or x is None or (isinstance(x, float) and np.isnan(x)) for x in diagnostic_true):
            return {'error': 'Invalid ground truth labels in diagnostic_true'}
        if any(x == -1 or x is None or (isinstance(x, float) and np.isnan(x)) for x in subtype_true):
            return {'error': 'Invalid ground truth labels in subtype_true'}
        
        # Diagnostic metrics
        diag_acc = accuracy_score(diagnostic_true, diagnostic_preds)
        diag_prec, diag_rec, diag_f1, _ = precision_recall_fscore_support(
            diagnostic_true, diagnostic_preds, average='weighted', zero_division=0
        )
        diag_cm = confusion_matrix(diagnostic_true, diagnostic_preds)
        
        # Subtype metrics
        sub_acc = accuracy_score(subtype_true, subtype_preds)
        sub_prec, sub_rec, sub_f1, _ = precision_recall_fscore_support(
            subtype_true, subtype_preds, average='weighted', zero_division=0
        )
        sub_cm = confusion_matrix(subtype_true, subtype_preds)
        
        # Per-class metrics
        diag_report = classification_report(
            diagnostic_true, diagnostic_preds,
            target_names=['Non-RD', 'RD'],
            output_dict=True,
            zero_division=0
        )
        
        sub_report = classification_report(
            subtype_true, subtype_preds,
            target_names=['Normal', 'PVD', 'Macula Intact', 'Macula Detached'],
            output_dict=True,
            zero_division=0
        )
        
        metrics = {
            'num_samples': len(valid_preds),
            'num_failed': len(predictions) - len(valid_preds),
            'diagnostic': {
                'accuracy': float(diag_acc),
                'precision': float(diag_prec),
                'recall': float(diag_rec),
                'f1_score': float(diag_f1),
                'confusion_matrix': diag_cm.tolist(),
                'per_class': diag_report
            },
            'subtype': {
                'accuracy': float(sub_acc),
                'precision': float(sub_prec),
                'recall': float(sub_rec),
                'f1_score': float(sub_f1),
                'confusion_matrix': sub_cm.tolist(),
                'per_class': sub_report
            }
        }
        
        return metrics
    
    except Exception as e:
        return {'error': f'Error computing metrics: {str(e)}'}


def save_confusion_matrices(metrics: Dict, output_dir: Path):
    """
    Save confusion matrix visualizations
    
    Args:
        metrics: Metrics dictionary
        output_dir: Output directory
    """
    # Check if metrics has error
    if 'error' in metrics:
        print(f"\nSkipping confusion matrices: {metrics['error']}")
        return
    
    cm_dir = output_dir / 'confusion_matrices'
    cm_dir.mkdir(parents=True, exist_ok=True)
    
    # Diagnostic confusion matrix
    diag_cm = np.array(metrics['diagnostic']['confusion_matrix'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        diag_cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Non-RD', 'RD'],
        yticklabels=['Non-RD', 'RD']
    )
    plt.title('Diagnostic Classification Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(cm_dir / 'diagnostic_confusion_matrix.png', dpi=150)
    plt.close()
    
    # Subtype confusion matrix
    sub_cm = np.array(metrics['subtype']['confusion_matrix'])
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        sub_cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Normal', 'PVD', 'Macula Intact', 'Macula Detached'],
        yticklabels=['Normal', 'PVD', 'Macula Intact', 'Macula Detached']
    )
    plt.title('Subtype Classification Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(cm_dir / 'subtype_confusion_matrix.png', dpi=150)
    plt.close()
    
    print(f"\nConfusion matrices saved to: {cm_dir}")


def save_results(
    predictions: List[Dict],
    metrics: Dict,
    output_dir: Path
):
    """
    Save all results to disk
    
    Args:
        predictions: List of predictions
        metrics: Evaluation metrics
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save predictions as CSV
    pred_df = pd.DataFrame(predictions)
    pred_csv_path = output_dir / 'test_set_predictions.csv'
    pred_df.to_csv(pred_csv_path, index=False)
    print(f"\nPredictions saved to: {pred_csv_path}")
    
    # Save metrics as JSON
    metrics_path = output_dir / 'test_set_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")
    
    # Save confusion matrices
    save_confusion_matrices(metrics, output_dir)
    
    # Print summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    
    # Check if metrics has error
    if 'error' in metrics:
        print(f"\nError: {metrics['error']}")
        print("="*80)
        return
    
    print(f"\nTotal samples: {metrics['num_samples']}")
    print(f"Failed samples: {metrics['num_failed']}")
    print(f"\nDiagnostic Classification:")
    print(f"  Accuracy:  {metrics['diagnostic']['accuracy']:.4f}")
    print(f"  Precision: {metrics['diagnostic']['precision']:.4f}")
    print(f"  Recall:    {metrics['diagnostic']['recall']:.4f}")
    print(f"  F1-Score:  {metrics['diagnostic']['f1_score']:.4f}")
    print(f"\nSubtype Classification:")
    print(f"  Accuracy:  {metrics['subtype']['accuracy']:.4f}")
    print(f"  Precision: {metrics['subtype']['precision']:.4f}")
    print(f"  Recall:    {metrics['subtype']['recall']:.4f}")
    print(f"  F1-Score:  {metrics['subtype']['f1_score']:.4f}")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Run batch inference on ERDES test set',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Required arguments
    parser.add_argument('--data_csv', type=str, required=True,
                       help='Path to balanced_split_desc.csv')
    parser.add_argument('--classifier_checkpoint', type=str, required=True,
                       help='Path to trained classifier checkpoint')
    
    # Optional arguments
    parser.add_argument('--vlm_checkpoint', type=str, default=None,
                       help='Path to finetuned VLM checkpoint (optional)')
    parser.add_argument('--output_dir', type=str, default='./test_set_inference_results',
                       help='Output directory for results')
    parser.add_argument('--video_base_dir', type=str, default=None,
                       help='Base directory for video files (if paths in CSV are relative)')
    
    # Model configuration
    parser.add_argument('--num_diagnostic_classes', type=int, default=2,
                       help='Number of diagnostic classes')
    parser.add_argument('--num_subtype_classes', type=int, default=4,
                       help='Number of subtype classes')
    parser.add_argument('--num_frames', type=int, default=32,
                       help='Number of frames to sample from video')
    parser.add_argument('--img_size', type=int, default=224,
                       help='Image size for preprocessing')
    
    # Processing options
    parser.add_argument('--num_workers', type=int, default=1,
                       help='Number of parallel workers (currently not used)')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for processing')
    parser.add_argument('--no_save_attention', action='store_true',
                       help='Do not save attention maps')
    parser.add_argument('--no_save_visualizations', action='store_true',
                       help='Do not generate visualizations')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Setup device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load data CSV
    print(f"\nLoading data from: {args.data_csv}")
    df = pd.read_csv(args.data_csv)
    
    # Check if CSV has 'split' column
    if 'split' in df.columns:
        # Use existing split - only process test samples
        test_df = df[df['split'] == 'test'].reset_index(drop=True)
        print(f"Found {len(test_df)} test samples from existing split")
    else:
        # No split column - process all rows
        test_df = df.copy()
        print(f"No 'split' column found, processing all {len(test_df)} samples")
        
        # Map diagnostic and subtype to numeric labels if they exist
        if 'diagnostic_class' in df.columns and df['diagnostic_class'].dtype == 'object':
            diagnostic_map = {'non_rd': 0, 'rd': 1}
            test_df['diagnostic_class'] = test_df['diagnostic_class'].map(diagnostic_map)
            print("Mapped diagnostic_class to numeric labels")
        
        if 'subtype' in df.columns and df['subtype'].dtype == 'object':
            subtype_map = {'normal': 0, 'pvd': 1, 'macula_intact': 2, 'macula_detached': 3}
            test_df['subtype_class'] = test_df['subtype'].map(subtype_map)
            print("Mapped subtype to numeric labels")
    
    if len(test_df) == 0:
        print("Error: No samples found in CSV")
        return
    
    # Load classifier
    print(f"\nLoading classifier from: {args.classifier_checkpoint}")
    model = create_multiclass_model(
        num_diagnostic_classes=args.num_diagnostic_classes,
        num_subtype_classes=args.num_subtype_classes,
        pretrained=False
    )
    
    checkpoint = torch.load(args.classifier_checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    print("Classifier loaded successfully")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run batch inference
    predictions, attention_maps = run_batch_inference(
        model=model,
        test_df=test_df,
        device=device,
        num_frames=args.num_frames,
        img_size=args.img_size,
        save_attention=not args.no_save_attention,
        output_dir=output_dir,
        video_base_dir=args.video_base_dir
    )
    
    # Compute metrics
    print("\nComputing evaluation metrics...")
    metrics = compute_metrics(predictions)
    
    # Save results
    save_results(predictions, metrics, output_dir)
    
    # TODO: VLM inference if checkpoint provided
    if args.vlm_checkpoint:
        print("\nNote: VLM inference on test set not yet implemented")
        print("      Only classifier predictions are generated")
    
    print(f"\nAll results saved to: {output_dir}")


if __name__ == '__main__':
    main()
