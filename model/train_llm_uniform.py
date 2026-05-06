#!/usr/bin/env python3
"""
Train VLM with Uniformly Sampled Frames (No Classifier, No Heatmaps)

This script trains Qwen 2.5 VL using uniformly sampled frames from videos,
without using classifier-based frame selection or heatmap overlays.
This serves as a baseline to compare against:
- Base model (classifier-selected frames, no heatmaps)
- FAVG model (classifier-selected frames with heatmaps and contrastive learning)
"""

import os
import argparse
import json
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
import pandas as pd

import torch
from tqdm import tqdm

from vlm_finetuning import MedicalVLMDataset, setup_qwen2vl_for_finetuning, train_vlm
from erdes_dataset import ERDESDataset


def extract_uniform_frames(video_path, num_frames=5, img_size=224):
    """
    Extract uniformly sampled frames from a video.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract
        img_size: Target image size
        
    Returns:
        List of frame arrays (RGB, uint8)
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < num_frames:
        # If video has fewer frames than requested, use all frames
        frame_indices = list(range(total_frames))
    else:
        # Uniformly sample frames
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize
        frame_resized = cv2.resize(frame_rgb, (img_size, img_size))
        
        frames.append(frame_resized)
    
    cap.release()
    
    return frames


def prepare_uniform_vlm_data(
    dataset,
    output_dir,
    num_frames=5,
    img_size=224
):
    """
    Prepare VLM training data using uniformly sampled frames.
    
    Args:
        dataset: ERDESDataset instance
        output_dir: Directory to save frames and metadata
        num_frames: Number of frames to extract per video
        img_size: Image size
        
    Returns:
        List of sample dictionaries
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    samples = []
    skipped_count = 0
    
    print(f"\nPreparing VLM data with uniform sampling...")
    print(f"Extracting {num_frames} frames per video")
    print(f"Output directory: {output_dir}")
    
    for idx in tqdm(range(len(dataset)), desc="Processing videos"):
        try:
            # Get video metadata
            video_info = dataset.df.iloc[idx]
            video_id = video_info['clip_id']
            video_path = Path(dataset.data_root) / video_info['file_path']
            
            # Skip if no summary
            if 'summary' not in video_info or pd.isna(video_info['summary']) or str(video_info['summary']).strip() == '':
                skipped_count += 1
                continue
            
            # Extract uniformly sampled frames
            frames = extract_uniform_frames(
                video_path=video_path,
                num_frames=num_frames,
                img_size=img_size
            )
            
            if len(frames) == 0:
                print(f"Warning: No frames extracted from {video_id}")
                skipped_count += 1
                continue
            
            # Save frames
            frame_paths = []
            for k, frame in enumerate(frames):
                frame_path = output_dir / f"{video_id}_uniform_frame_{k}.jpg"
                cv2.imwrite(str(frame_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                frame_paths.append(str(frame_path))
            
            # Create prompt (simple, no classifier predictions)
            prompt = "Analyze this ocular ultrasound video and provide a detailed clinical diagnosis."
            
            # Get ground truth
            ground_truth = video_info['summary']
            
            # Create sample
            sample = {
                'video_id': video_id,
                'frame_paths': frame_paths,
                'prompt': prompt,
                'ground_truth': ground_truth,
                'num_frames': len(frames),
                'sampling_method': 'uniform',
                'diagnosis_text': video_info.get('diagnosis_text', ''),
                'diagnostic_class': video_info.get('diagnostic_class', ''),
                'subtype': video_info.get('subtype', '')
            }
            
            samples.append(sample)
            
        except Exception as e:
            print(f"\nError processing video {idx}: {str(e)}")
            import traceback
            traceback.print_exc()
            skipped_count += 1
            continue
    
    # Save all samples metadata
    samples_file = output_dir / 'all_samples.json'
    with open(samples_file, 'w') as f:
        json.dump(samples, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"VLM Data Preparation Complete (Uniform Sampling)")
    print(f"{'='*60}")
    print(f"Prepared samples: {len(samples)}")
    if skipped_count > 0:
        print(f"Skipped (no summary or error): {skipped_count}")
    print(f"Saved to: {samples_file}")
    print(f"{'='*60}\n")
    
    return samples


def check_cache_validity(output_dir: Path, split: str = 'train') -> bool:
    """
    Check if cached VLM data exists and is valid.
    
    Args:
        output_dir: Output directory path
        split: 'train' or 'test'
        
    Returns:
        True if valid cache exists, False otherwise
    """
    cache_dir = output_dir / 'vlm_data' / split
    samples_file = cache_dir / 'all_samples.json'
    
    # Check if samples file exists
    if not samples_file.exists():
        return False
    
    # Load and validate samples
    try:
        with open(samples_file, 'r') as f:
            samples = json.load(f)
        
        if not samples or len(samples) == 0:
            print(f"Warning: {split} samples file is empty")
            return False
        
        # Check if all samples have required keys
        for sample in samples[:10]:  # Check first 10 samples
            if 'frame_paths' not in sample:
                print(f"Warning: {split} cache is missing frame_paths")
                return False
        
        # Check if all referenced files exist
        missing_files = []
        for sample in samples[:5]:  # Check first 5 samples
            if 'frame_paths' in sample:
                for file_path in sample['frame_paths']:
                    if not Path(file_path).exists():
                        missing_files.append(file_path)
                        break
                if missing_files:
                    break
        
        if missing_files:
            print(f"Warning: Some {split} data files are missing: {missing_files[0]}")
            return False
        
        print(f"✓ Found valid {split} cache with {len(samples)} samples")
        return True
        
    except (json.JSONDecodeError, KeyError, Exception) as e:
        print(f"Warning: Error reading {split} cache: {e}")
        return False


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config = vars(args)
    config['timestamp'] = datetime.now().strftime('%Y%m%d_%H%M%S')
    config['sampling_method'] = 'uniform'
    config['use_classifier'] = False
    config['use_heatmaps'] = False
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n" + "="*60)
    print("VLM Training Pipeline - Uniform Sampling")
    print("="*60)
    print(f"Train CSV: {args.train_csv}")
    print(f"Test CSV: {args.test_csv}")
    print(f"Data Root: {args.data_root}")
    print(f"Output Dir: {output_dir}")
    print(f"VLM Model: {args.vlm_model}")
    print(f"Sampling: Uniform ({args.num_frames} frames per video)")
    print(f"Heatmaps: No")
    print(f"Classifier: No")
    print("="*60 + "\n")
    
    # Load datasets
    print("Loading train dataset...")
    train_dataset = ERDESDataset(
        csv_path=args.train_csv,
        data_root=args.data_root,
        num_frames=32,  # Not used for uniform sampling, but required by dataset
        img_size=args.img_size
    )
    
    print("Loading test dataset...")
    test_dataset = ERDESDataset(
        csv_path=args.test_csv,
        data_root=args.data_root,
        num_frames=32,
        img_size=args.img_size
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Define data directories
    train_data_dir = output_dir / 'vlm_data' / 'train'
    test_data_dir = output_dir / 'vlm_data' / 'test'
    
    # Check cache validity (unless force_prepare is set)
    if args.force_prepare:
        print("\n--force_prepare flag set: Will re-generate all data")
        train_cache_valid = False
        test_cache_valid = False
    else:
        train_cache_valid = check_cache_validity(output_dir, 'train')
        test_cache_valid = check_cache_validity(output_dir, 'test')
    
    # Prepare VLM data (or use cache)
    use_cache = (args.skip_data_preparation or not args.force_prepare) and train_cache_valid and test_cache_valid
    
    if use_cache:
        print("\n" + "="*60)
        print("Step 1: Using Cached VLM Data")
        print("="*60)
        print("✓ Skipping data preparation (valid cache found)")
        print(f"   Train cache: {train_data_dir / 'all_samples.json'}")
        print(f"   Test cache: {test_data_dir / 'all_samples.json'}")
        print("\nTip: Use --force_prepare to regenerate data")
    else:
        print("\n" + "="*60)
        print("Step 1: Preparing VLM Training Data (Uniform Sampling)")
        print("="*60)
        
        if not train_cache_valid or args.force_prepare:
            if args.force_prepare and train_cache_valid:
                print("\nForce re-preparing train data (cache exists but --force_prepare set)...")
            else:
                print("\nPreparing train data...")
            train_samples = prepare_uniform_vlm_data(
                dataset=train_dataset,
                output_dir=train_data_dir,
                num_frames=args.num_frames,
                img_size=args.img_size
            )
        else:
            print("✓ Using cached train data")
        
        if not test_cache_valid or args.force_prepare:
            if args.force_prepare and test_cache_valid:
                print("\nForce re-preparing test data (cache exists but --force_prepare set)...")
            else:
                print("\nPreparing test data...")
            test_samples = prepare_uniform_vlm_data(
                dataset=test_dataset,
                output_dir=test_data_dir,
                num_frames=args.num_frames,
                img_size=args.img_size
            )
        else:
            print("✓ Using cached test data")
    
    # Setup VLM
    print("\n" + "="*60)
    print("Step 2: Setting up VLM Model")
    print("="*60)
    
    model, processor = setup_qwen2vl_for_finetuning(
        model_name=args.vlm_model,
        load_in_4bit=args.use_4bit,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )
    
    print("✓ VLM model setup complete!")
    
    # Create VLM datasets
    print("\nCreating VLM datasets...")
    print(f"Using original frames (no heatmaps)")
    
    train_vlm_dataset = MedicalVLMDataset(
        samples_json=str(train_data_dir / 'all_samples.json'),
        processor=processor,
        use_heatmaps=False  # Always use original frames
    )
    
    test_vlm_dataset = MedicalVLMDataset(
        samples_json=str(test_data_dir / 'all_samples.json'),
        processor=processor,
        use_heatmaps=False
    )
    
    print(f"Train VLM samples: {len(train_vlm_dataset)}")
    print(f"Test VLM samples: {len(test_vlm_dataset)}")
    
    # Train VLM
    print("\n" + "="*60)
    print("Step 3: Training VLM")
    print("="*60)
    
    vlm_output_dir = output_dir / 'vlm_checkpoints'
    
    trained_model = train_vlm(
        model=model,
        processor=processor,
        train_dataset=train_vlm_dataset,
        val_dataset=test_vlm_dataset,
        output_dir=str(vlm_output_dir),
        num_epochs=args.vlm_epochs,
        batch_size=args.vlm_batch_size,
        learning_rate=args.vlm_lr,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps
    )
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"VLM checkpoints saved to: {vlm_output_dir}")
    print(f"Best model: {vlm_output_dir / 'best_model'}")
    print("\n📝 Training Configuration:")
    print(f"  ✓ Uniform frame sampling ({args.num_frames} frames per video)")
    print(f"  ✓ No classifier-based frame selection")
    print(f"  ✓ No heatmap overlays")
    print(f"  ✓ No contrastive learning")
    print("="*60 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train VLM with Uniform Frame Sampling')
    
    # Data arguments
    parser.add_argument('--train_csv', type=str,
                       default='../benchmarks/input/balanced_split_desc_train.csv',
                       help='Path to training CSV file')
    parser.add_argument('--test_csv', type=str,
                       default='../benchmarks/input/balanced_split_desc_test.csv',
                       help='Path to test CSV file')
    parser.add_argument('--data_root', type=str,
                       default='../erdes',
                       help='Root directory for video data')
    parser.add_argument('--output_dir', type=str,
                       default='./checkpoints/vlm_uniform',
                       help='Output directory for VLM checkpoints')
    
    # Data preparation arguments
    parser.add_argument('--skip_data_preparation', action='store_true',
                       help='Skip data preparation if cache exists')
    parser.add_argument('--force_prepare', action='store_true',
                       help='Force re-preparation even if cache exists')
    parser.add_argument('--num_frames', type=int, default=5,
                       help='Number of frames to uniformly sample from each video')
    parser.add_argument('--img_size', type=int, default=224,
                       help='Image size (height and width)')
    
    # VLM model arguments
    parser.add_argument('--vlm_model', type=str,
                       default='Qwen/Qwen2-VL-7B-Instruct',
                       help='VLM model name or path')
    parser.add_argument('--use_4bit', action='store_true', default=True,
                       help='Use 4-bit quantization')
    
    # LoRA arguments
    parser.add_argument('--lora_r', type=int, default=16,
                       help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32,
                       help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05,
                       help='LoRA dropout')
    
    # Training arguments
    parser.add_argument('--vlm_epochs', type=int, default=10,
                       help='Number of VLM training epochs')
    parser.add_argument('--vlm_batch_size', type=int, default=2,
                       help='VLM batch size')
    parser.add_argument('--vlm_lr', type=float, default=2e-5,
                       help='VLM learning rate')
    parser.add_argument('--warmup_steps', type=int, default=100,
                       help='Number of warmup steps')
    parser.add_argument('--logging_steps', type=int, default=10,
                       help='Logging frequency')
    parser.add_argument('--eval_steps', type=int, default=100,
                       help='Evaluation frequency')
    parser.add_argument('--save_steps', type=int, default=100,
                       help='Checkpoint save frequency')
    
    args = parser.parse_args()
    
    main(args)
