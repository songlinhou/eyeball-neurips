#!/usr/bin/env python3
"""
Train VLM (Vision-Language Model) for Medical Video Diagnosis

This script:
1. Loads a pretrained multiclass classifier
2. Prepares VLM training data with important frames and heatmaps
3. Finetunes Qwen 2.5 VL with LoRA
4. Saves the best VLM checkpoint
"""

import os
import argparse
import json
from pathlib import Path
from datetime import datetime
import shutil

import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from multiclass_model import create_multiclass_model
from vlm_data_preparation import VLMDataPreparator
from vlm_finetuning import MedicalVLMDataset, setup_qwen2vl_for_finetuning, train_vlm
from erdes_dataset import ERDESDataset, collate_fn


def get_balanced_splits(dataset, test_size=0.2, random_state=42, cache_file=None):
    """
    Create balanced train/test splits with caching
    
    Args:
        dataset: Full dataset
        test_size: Fraction for test set
        random_state: Random seed
        cache_file: Path to cache file (if None, no caching)
        
    Returns:
        train_indices, test_indices
    """
    # Try to load from cache
    if cache_file and Path(cache_file).exists():
        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            
            # Validate cache matches current settings
            if (cache_data.get('test_size') == test_size and 
                cache_data.get('random_state') == random_state and
                cache_data.get('dataset_size') == len(dataset)):
                
                train_indices = cache_data['train_indices']
                test_indices = cache_data['test_indices']
                print(f"✓ Loaded splits from cache: {len(train_indices)} train, {len(test_indices)} test")
                return train_indices, test_indices
            else:
                print("Cache exists but parameters don't match, regenerating splits...")
        except (json.JSONDecodeError, KeyError, Exception) as e:
            print(f"Warning: Could not load split cache: {e}")
    
    # Generate splits
    print("Generating balanced train/test splits...")
    
    # Get all labels
    diagnostic_labels = []
    subtype_labels = []
    
    for idx in range(len(dataset)):
        _, labels, _ = dataset[idx]
        diagnostic_labels.append(labels['diagnostic'])
        subtype_labels.append(labels['subtype'])
    
    # Create stratification key
    stratify_labels = [f"{d}_{s}" for d, s in zip(diagnostic_labels, subtype_labels)]
    
    # Split with stratification
    indices = list(range(len(dataset)))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        stratify=stratify_labels,
        random_state=random_state
    )
    
    # Save to cache
    if cache_file:
        cache_data = {
            'test_size': test_size,
            'random_state': random_state,
            'dataset_size': len(dataset),
            'train_indices': train_indices,
            'test_indices': test_indices,
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
        }
        
        Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
        print(f"✓ Saved splits to cache: {cache_file}")
    
    return train_indices, test_indices


def prepare_vlm_data(classifier, dataset, output_dir, device, top_k_frames=5, use_contrastive=True):
    """
    Prepare VLM training data from videos
    
    Args:
        classifier: Trained multiclass model
        dataset: ERDESDataset or Subset
        output_dir: Directory to save prepared data
        device: Device for inference
        top_k_frames: Number of important frames to extract
        use_contrastive: Whether to create contrastive samples
        
    Returns:
        List of prepared sample metadata
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize data preparator
    preparator = VLMDataPreparator(
        model=classifier,
        device=device,
        top_k_frames=top_k_frames
    )
    
    samples = []
    skipped_count = 0
    
    print("\nPreparing VLM data...")
    for idx in tqdm(range(len(dataset)), desc="Processing videos"):
        # Get video and metadata
        if isinstance(dataset, Subset):
            video, labels, metadata = dataset.dataset[dataset.indices[idx]]
        else:
            video, labels, metadata = dataset[idx]
        
        video_id = metadata['clip_id']
        video_tensor = video.unsqueeze(0)  # Add batch dimension
        
        # Validate video tensor shape
        if video_tensor.dim() != 5:
            print(f"\nWarning: Unexpected video tensor shape for {video_id}: {video_tensor.shape}")
            print(f"Expected 5D tensor (B, C, T, H, W), got {video_tensor.dim()}D")
            continue
        
        if video_tensor.shape[2] == 0:  # No temporal frames
            print(f"\nWarning: Video {video_id} has 0 temporal frames, skipping")
            continue
        
        # Check if summary exists - skip videos without ground truth
        summary = metadata.get('summary', '').strip()
        if not summary:
            skipped_count += 1
            continue
        
        # Prepare ground truth (include summary and diagnosis_text from CSV)
        ground_truth = {
            'diagnostic': metadata['diagnostic_class'],
            'subtype': metadata['subtype'],
            'anatomical': metadata.get('anatomical_subclass', 'N/A'),
            'summary': summary,  # Expert clinical summary from CSV
            'diagnosis_text': metadata.get('diagnosis_text', '')  # Structured diagnosis from CSV
        }
        
        # Create sample directory
        sample_dir = output_dir / video_id
        sample_dir.mkdir(exist_ok=True)
        
        try:
            # Prepare VLM sample
            sample = preparator.prepare_vlm_sample(
                video_tensor=video_tensor,
                video_id=video_id,
                output_dir=str(sample_dir),
                ground_truth=ground_truth
            )
            
            samples.append(sample)
            
            # Optionally create contrastive samples
            if use_contrastive:
                correct_sample, contrastive_sample = preparator.create_contrastive_samples(
                    video_tensor=video_tensor,
                    video_id=video_id,
                    output_dir=str(sample_dir),
                    ground_truth=ground_truth
                )
                samples.append(correct_sample)
                samples.append(contrastive_sample)
                
        except Exception as e:
            import traceback
            print(f"\nError processing {video_id}: {e}")
            print("Full traceback:")
            traceback.print_exc()
            continue
    
    # Save all samples metadata
    samples_file = output_dir / 'all_samples.json'
    with open(samples_file, 'w') as f:
        json.dump(samples, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"VLM Data Preparation Complete")
    print(f"{'='*60}")
    print(f"Prepared samples: {len(samples)}")
    if skipped_count > 0:
        print(f"Skipped (no summary): {skipped_count}")
    print(f"Saved to: {samples_file}")
    print(f"{'='*60}\n")
    
    return samples


def check_cache_validity(output_dir: Path, split: str = 'train') -> bool:
    """
    Check if cached VLM data exists and is valid
    
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
        
        # Check if all referenced files exist
        missing_files = []
        for sample in samples[:5]:  # Check first 5 samples
            for path_key in ['frame_paths', 'heatmap_paths']:
                if path_key in sample:
                    for file_path in sample[path_key]:
                        if not Path(file_path).exists():
                            missing_files.append(file_path)
                            break
                    if missing_files:
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
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n" + "="*60)
    print("VLM Training Pipeline")
    print("="*60)
    print(f"Classifier Checkpoint: {args.classifier_checkpoint}")
    print(f"Train CSV: {args.train_csv}")
    print(f"Test CSV: {args.test_csv}")
    print(f"Data Root: {args.data_root}")
    print(f"Output Dir: {output_dir}")
    print(f"VLM Model: {args.vlm_model}")
    print("="*60 + "\n")
    
    # Load classifier
    print("Loading pretrained classifier...")
    classifier = create_multiclass_model(
        num_diagnostic_classes=args.num_diagnostic_classes,
        num_subtype_classes=args.num_subtype_classes,
        pretrained=False
    )
    
    # Load checkpoint
    checkpoint = torch.load(args.classifier_checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        classifier.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"Test accuracy: {checkpoint.get('test_acc', 'unknown')}")
    else:
        classifier.load_state_dict(checkpoint)
    
    classifier = classifier.to(device)
    classifier.eval()
    print("✓ Classifier loaded successfully!")
    
    # Load datasets
    print("\nLoading train dataset...")
    train_dataset = ERDESDataset(
        csv_path=args.train_csv,
        data_root=args.data_root,
        num_frames=args.num_frames,
        img_size=args.img_size
    )
    
    print("Loading test dataset...")
    test_dataset = ERDESDataset(
        csv_path=args.test_csv,
        data_root=args.data_root,
        num_frames=args.num_frames,
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
        print("Step 1: Preparing VLM Training Data")
        print("="*60)
        
        if not train_cache_valid or args.force_prepare:
            if args.force_prepare and train_cache_valid:
                print("\nForce re-preparing train data (cache exists but --force_prepare set)...")
            else:
                print("\nPreparing train data...")
            train_samples = prepare_vlm_data(
                classifier=classifier,
                dataset=train_dataset,
                output_dir=train_data_dir,
                device=device,
                top_k_frames=args.top_k_frames,
                use_contrastive=args.use_contrastive
            )
        else:
            print("✓ Using cached train data")
        
        if not test_cache_valid or args.force_prepare:
            if args.force_prepare and test_cache_valid:
                print("\nForce re-preparing test data (cache exists but --force_prepare set)...")
            else:
                print("\nPreparing test data...")
            test_samples = prepare_vlm_data(
                classifier=classifier,
                dataset=test_dataset,
                output_dir=test_data_dir,
                device=device,
                top_k_frames=args.top_k_frames,
                use_contrastive=False  # No contrastive samples for test
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
    train_vlm_dataset = MedicalVLMDataset(
        samples_json=str(train_data_dir / 'all_samples.json'),
        processor=processor
    )
    
    test_vlm_dataset = MedicalVLMDataset(
        samples_json=str(test_data_dir / 'all_samples.json'),
        processor=processor
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
    print("="*60 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train VLM for Medical Video Diagnosis')
    
    # Classifier arguments
    parser.add_argument('--classifier_checkpoint', type=str, required=True,
                       help='Path to trained classifier checkpoint (.pth file)')
    parser.add_argument('--num_diagnostic_classes', type=int, default=2,
                       help='Number of diagnostic classes')
    parser.add_argument('--num_subtype_classes', type=int, default=4,
                       help='Number of subtype classes')
    
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
                       default='./checkpoints/vlm',
                       help='Output directory for VLM checkpoints')
    
    # Legacy arguments (kept for backward compatibility, but ignored)
    parser.add_argument('--csv_path', type=str, default=None,
                       help='(Deprecated) Use --train_csv and --test_csv instead')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='(Deprecated) No longer used with separate train/test CSVs')
    parser.add_argument('--random_state', type=int, default=42,
                       help='(Deprecated) No longer used with separate train/test CSVs')
    
    # Data preparation arguments
    parser.add_argument('--skip_data_preparation', action='store_true',
                       help='Skip data preparation if cache exists (default: auto-detect)')
    parser.add_argument('--force_prepare', action='store_true',
                       help='Force re-preparation even if cache exists')
    parser.add_argument('--top_k_frames', type=int, default=5,
                       help='Number of important frames to extract')
    parser.add_argument('--use_contrastive', action='store_true', default=True,
                       help='Create contrastive samples for training')
    parser.add_argument('--num_frames', type=int, default=32,
                       help='Number of frames to sample from video')
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
