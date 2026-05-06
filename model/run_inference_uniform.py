#!/usr/bin/env python3
"""
Inference Script for Uniformly-Trained VLM Model

This script runs inference on the test set using a VLM model trained with
uniformly sampled frames (no classifier, no heatmaps).

Usage:
    python run_inference_uniform.py \
        --vlm_checkpoint ./checkpoints/vlm_uniform/vlm_checkpoints/best_model \
        --test_csv ../benchmarks/input/balanced_split_desc_test.csv \
        --data_root ../erdes \
        --output_dir ./results/uniform_inference \
        --num_frames 5

Features:
    - Uniformly samples N frames from each test video
    - No classifier needed
    - No heatmap overlays
    - Simple prompt without predictions
    - Generates predictions CSV for comparison
"""

import torch
import argparse
import json
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

from vlm_finetuning import setup_qwen2vl_for_finetuning, inference_vlm
from train_llm_uniform import extract_uniform_frames


def load_vlm_model(checkpoint_path: str, device: str = 'cuda'):
    """
    Load the uniformly-trained VLM model
    
    Args:
        checkpoint_path: Path to VLM checkpoint directory
        device: Device to load model on
        
    Returns:
        model: Loaded VLM model
        processor: VLM processor
    """
    print(f"Loading VLM model from {checkpoint_path}...")
    
    # Load model and processor
    model, processor = setup_qwen2vl_for_finetuning(
        model_name=checkpoint_path,
        load_in_4bit=True,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05
    )
    
    model.eval()
    print("✓ VLM model loaded successfully!")
    
    return model, processor


def prepare_sample_for_inference(
    video_path: Path,
    num_frames: int = 5,
    img_size: int = 224
) -> Dict:
    """
    Prepare a single video sample for VLM inference
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to uniformly sample
        img_size: Image size
        
    Returns:
        sample: Dictionary with frame paths and prompt
    """
    # Extract uniformly sampled frames
    frames = extract_uniform_frames(
        video_path=str(video_path),
        num_frames=num_frames,
        img_size=img_size
    )
    
    if len(frames) == 0:
        return None
    
    # Create simple prompt (no classifier predictions)
    prompt = "Analyze this ocular ultrasound video and provide a detailed clinical diagnosis."
    
    # Create sample dictionary
    sample = {
        'frame_paths': frames,  # List of numpy arrays
        'prompt': prompt
    }
    
    return sample


def run_vlm_inference(
    model,
    processor,
    frames: List[np.ndarray],
    prompt: str,
    max_new_tokens: int = 512
) -> str:
    """
    Run VLM inference on uniformly sampled frames
    
    Args:
        model: VLM model
        processor: VLM processor
        frames: List of frame arrays (RGB, uint8)
        prompt: Text prompt
        max_new_tokens: Maximum tokens to generate
        
    Returns:
        generated_text: VLM prediction
    """
    # Prepare messages for Qwen2VL
    messages = [
        {
            "role": "user",
            "content": [
                *[{"type": "image", "image": frame} for frame in frames],
                {"type": "text", "text": prompt}
            ]
        }
    ]
    
    # Process inputs
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Prepare image inputs
    from PIL import Image
    image_inputs = [Image.fromarray(frame) for frame in frames]
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        return_tensors="pt",
        padding=True
    )
    
    # Move to device
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v 
              for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0
        )
    
    # Decode
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
    
    return output_text


def run_inference_on_test_set(
    vlm_checkpoint: str,
    test_csv: str,
    data_root: str,
    output_dir: str,
    num_frames: int = 5,
    img_size: int = 224,
    max_new_tokens: int = 512
):
    """
    Run inference on entire test set
    
    Args:
        vlm_checkpoint: Path to VLM checkpoint
        test_csv: Path to test CSV file
        data_root: Root directory for video data
        output_dir: Output directory for results
        num_frames: Number of frames to uniformly sample
        img_size: Image size
        max_new_tokens: Maximum tokens to generate
    """
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("VLM Inference - Uniform Sampling")
    print("="*80)
    print(f"VLM Checkpoint: {vlm_checkpoint}")
    print(f"Test CSV: {test_csv}")
    print(f"Data Root: {data_root}")
    print(f"Output Dir: {output_dir}")
    print(f"Frames per video: {num_frames}")
    print(f"Device: {device}")
    print("="*80 + "\n")
    
    # Load VLM model
    model, processor = load_vlm_model(vlm_checkpoint, device)
    
    # Load test data
    print("Loading test data...")
    df = pd.read_csv(test_csv)
    print(f"Test samples: {len(df)}")
    
    # Run inference
    results = []
    skipped = 0
    
    print("\nRunning inference on test set...")
    for idx in tqdm(range(len(df)), desc="Processing videos"):
        try:
            row = df.iloc[idx]
            video_id = row['clip_id']
            video_path = Path(data_root) / row['file_path']
            
            # Check if video exists
            if not video_path.exists():
                print(f"\nWarning: Video not found: {video_path}")
                skipped += 1
                continue
            
            # Extract uniformly sampled frames
            frames = extract_uniform_frames(
                video_path=str(video_path),
                num_frames=num_frames,
                img_size=img_size
            )
            
            if len(frames) == 0:
                print(f"\nWarning: No frames extracted from {video_id}")
                skipped += 1
                continue
            
            # Create prompt
            prompt = "Analyze this ocular ultrasound video and provide a detailed clinical diagnosis."
            
            # Run VLM inference
            prediction = run_vlm_inference(
                model=model,
                processor=processor,
                frames=frames,
                prompt=prompt,
                max_new_tokens=max_new_tokens
            )
            
            # Store result
            result = {
                'clip_id': video_id,
                'filepath': row['file_path'],
                'predicted_summary': prediction,
                'ground_truth_summary': row.get('summary', ''),
                'ground_truth_diagnosis': row.get('diagnosis_text', ''),
                'diagnostic_class': row.get('diagnostic_class', ''),
                'subtype': row.get('subtype', ''),
                'anatomical_subclass': row.get('anatomical_subclass', ''),
                'num_frames_used': len(frames),
                'sampling_method': 'uniform'
            }
            
            results.append(result)
            
        except Exception as e:
            print(f"\nError processing {idx}: {str(e)}")
            import traceback
            traceback.print_exc()
            skipped += 1
            continue
    
    # Save results
    print(f"\n{'='*80}")
    print("Saving results...")
    
    # Save as CSV (compatible with llm_as_judge.py)
    results_df = pd.DataFrame(results)
    csv_path = output_dir / 'uniform_predictions.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"✓ Saved predictions CSV: {csv_path}")
    
    # Save as JSON
    json_path = output_dir / 'uniform_predictions.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved predictions JSON: {json_path}")
    
    # Save summary statistics
    summary = {
        'total_samples': len(df),
        'successful_predictions': len(results),
        'skipped': skipped,
        'num_frames_per_video': num_frames,
        'sampling_method': 'uniform',
        'vlm_checkpoint': vlm_checkpoint
    }
    
    summary_path = output_dir / 'inference_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved summary: {summary_path}")
    
    print(f"\n{'='*80}")
    print("Inference Complete!")
    print(f"{'='*80}")
    print(f"Total samples: {len(df)}")
    print(f"Successful predictions: {len(results)}")
    print(f"Skipped: {skipped}")
    print(f"\nResults saved to: {output_dir}")
    print(f"\nTo compare with other models using GPT-as-judge:")
    print(f"  cd ../benchmarks")
    print(f"  python llm_as_judge.py \\")
    print(f"    --predictions-1 ../model/{output_dir}/uniform_predictions.csv \\")
    print(f"    --predictions-2 output/FAVG_pred.csv \\")
    print(f"    --model-1-name 'Uniform Sampling' \\")
    print(f"    --model-2-name 'FAVG Model'")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run inference on test set using uniformly-trained VLM'
    )
    
    # Model arguments
    parser.add_argument('--vlm_checkpoint', type=str, required=True,
                       help='Path to VLM checkpoint directory')
    
    # Data arguments
    parser.add_argument('--test_csv', type=str,
                       default='../benchmarks/input/balanced_split_desc_test.csv',
                       help='Path to test CSV file')
    parser.add_argument('--data_root', type=str,
                       default='../erdes',
                       help='Root directory for video data')
    parser.add_argument('--output_dir', type=str,
                       default='./results/uniform_inference',
                       help='Output directory for results')
    
    # Inference arguments
    parser.add_argument('--num_frames', type=int, default=5,
                       help='Number of frames to uniformly sample (should match training)')
    parser.add_argument('--img_size', type=int, default=224,
                       help='Image size')
    parser.add_argument('--max_new_tokens', type=int, default=512,
                       help='Maximum tokens to generate')
    
    args = parser.parse_args()
    
    # Run inference
    run_inference_on_test_set(
        vlm_checkpoint=args.vlm_checkpoint,
        test_csv=args.test_csv,
        data_root=args.data_root,
        output_dir=args.output_dir,
        num_frames=args.num_frames,
        img_size=args.img_size,
        max_new_tokens=args.max_new_tokens
    )
