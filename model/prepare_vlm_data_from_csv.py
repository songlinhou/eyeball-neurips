"""
Prepare VLM training data from CSV file with ground-truth summaries
Reads balanced_split_desc.csv and creates JSON samples with expert summaries
"""

import pandas as pd
import json
from pathlib import Path
from typing import List, Dict
import argparse


def prepare_vlm_samples_from_csv(
    csv_path: str,
    classifier_predictions_json: str,
    output_json: str,
    heatmap_dir: str = None
):
    """
    Prepare VLM training samples by combining CSV summaries with classifier predictions
    
    Args:
        csv_path: Path to balanced_split_desc.csv with ground-truth summaries
        classifier_predictions_json: JSON file with classifier predictions and attention maps
        output_json: Output JSON file for VLM training
        heatmap_dir: Directory containing heatmap overlay images
    """
    # Load CSV with ground-truth summaries
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples from CSV")
    
    # Load classifier predictions
    with open(classifier_predictions_json, 'r') as f:
        predictions = json.load(f)
    
    # Create mapping from clip_id to summary and diagnosis_text
    summary_map = dict(zip(df['clip_id'], df['summary']))
    diagnosis_text_map = dict(zip(df['clip_id'], df['diagnosis_text']))
    diagnostic_map = dict(zip(df['clip_id'], df['diagnostic_class']))
    subtype_map = dict(zip(df['clip_id'], df['subtype']))
    
    samples = []
    
    for pred in predictions:
        video_id = pred['video_id']
        clip_id = video_id.split('/')[-1].replace('.mp4', '')
        
        # Get ground-truth summary from CSV
        if clip_id not in summary_map:
            print(f"Warning: {clip_id} not found in CSV, skipping...")
            continue
        
        summary = summary_map[clip_id]
        diagnosis_text = diagnosis_text_map[clip_id]
        gt_diagnostic = diagnostic_map[clip_id]
        gt_subtype = subtype_map[clip_id]
        
        # Create prompt with predictions
        prompt = f"""The AI model predicts:
- Primary Diagnosis: {pred['predicted_diagnostic']} (confidence: {pred['diagnostic_confidence']:.1%})
- Subtype: {pred['predicted_subtype']} (confidence: {pred['subtype_confidence']:.1%})

Based on the highlighted regions in these ultrasound frames, explain the clinical reasoning."""
        
        # Prepare sample
        sample = {
            "video_id": f"{clip_id}_correct",
            "frame_paths": pred['important_frame_paths'],
            "prompt": prompt,
            "predictions": {
                "diagnostic": pred['predicted_diagnostic'],
                "diagnostic_confidence": pred['diagnostic_confidence'],
                "subtype": pred['predicted_subtype'],
                "subtype_confidence": pred['subtype_confidence']
            },
            "summary": summary,  # Ground truth clinical description from CSV
            "diagnosis_text": diagnosis_text,  # Structured diagnosis from CSV
            "ground_truth": {
                "diagnostic": gt_diagnostic,
                "subtype": gt_subtype
            },
            "is_contrastive": False
        }
        
        # Add heatmap paths if available
        if heatmap_dir and 'attention_maps' in pred:
            heatmap_paths = []
            for i, frame_path in enumerate(pred['important_frame_paths']):
                frame_name = Path(frame_path).stem
                heatmap_path = Path(heatmap_dir) / f"{clip_id}_frame{i}_heatmap.jpg"
                if heatmap_path.exists():
                    heatmap_paths.append(str(heatmap_path))
                else:
                    heatmap_paths.append(frame_path)  # Fallback to original
            sample['heatmap_paths'] = heatmap_paths
        
        samples.append(sample)
        
        # Create contrastive sample with spatially-shifted heatmap
        if heatmap_dir:
            contrastive_sample = sample.copy()
            contrastive_sample['video_id'] = f"{clip_id}_contrastive"
            contrastive_sample['is_contrastive'] = True
            # Heatmap paths would point to shifted versions (created separately)
            if 'heatmap_paths' in sample:
                contrastive_sample['heatmap_paths'] = [
                    p.replace('_heatmap.jpg', '_heatmap_shifted.jpg') 
                    for p in sample['heatmap_paths']
                ]
            samples.append(contrastive_sample)
    
    # Save samples
    with open(output_json, 'w') as f:
        json.dump(samples, f, indent=2)
    
    print(f"Created {len(samples)} samples ({len([s for s in samples if not s['is_contrastive']])} correct, "
          f"{len([s for s in samples if s['is_contrastive']])} contrastive)")
    print(f"Saved to {output_json}")


def create_example_sample():
    """Create an example sample to show the format"""
    example = {
        "video_id": "164267_02030_correct",
        "frame_paths": [
            "frames/164267_02030_frame0.jpg",
            "frames/164267_02030_frame1.jpg",
            "frames/164267_02030_frame2.jpg",
            "frames/164267_02030_frame3.jpg",
            "frames/164267_02030_frame4.jpg"
        ],
        "heatmap_paths": [
            "heatmaps/164267_02030_frame0_heatmap.jpg",
            "heatmaps/164267_02030_frame1_heatmap.jpg",
            "heatmaps/164267_02030_frame2_heatmap.jpg",
            "heatmaps/164267_02030_frame3_heatmap.jpg",
            "heatmaps/164267_02030_frame4_heatmap.jpg"
        ],
        "prompt": """The AI model predicts:
- Primary Diagnosis: non_rd (confidence: 95.2%)
- Subtype: normal (confidence: 92.3%)

Based on the highlighted regions in these ultrasound frames, explain the clinical reasoning.""",
        "predictions": {
            "diagnostic": "non_rd",
            "diagnostic_confidence": 0.952,
            "subtype": "normal",
            "subtype_confidence": 0.923
        },
        "summary": "The video consistently reveals a normal ocular structure across all frames, characterized by a smoothly contoured, continuous retinal line closely adhered to the posterior wall of the eye globe, indicative of no retinal detachment. There are no signs of mobile membranes, vitreous abnormalities, or additional membranous structures, pointing to the absence of retinal pathology. The well-defined globe maintains a regular shape throughout, with uniformly anechoic vitreous content, further supporting the conclusion of normal ocular health. Overall, the ultrasound video displays no pathological alterations, confirming the integrity of the retina and the anatomical normalcy of the eye.",
        "diagnosis_text": "<diagnostic>non_rd</diagnostic><subtype>normal</subtype><anatomical>nan</anatomical>",
        "ground_truth": {
            "diagnostic": "non_rd",
            "subtype": "normal"
        },
        "is_contrastive": False
    }
    
    return example


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare VLM training data from CSV')
    parser.add_argument('--csv_path', type=str, 
                       default='benchmarks/input/balanced_split_desc.csv',
                       help='Path to CSV with ground-truth summaries')
    parser.add_argument('--predictions_json', type=str,
                       required=True,
                       help='JSON file with classifier predictions')
    parser.add_argument('--output_json', type=str,
                       default='vlm_training_samples.json',
                       help='Output JSON file')
    parser.add_argument('--heatmap_dir', type=str,
                       help='Directory containing heatmap images')
    parser.add_argument('--show_example', action='store_true',
                       help='Show example sample format')
    
    args = parser.parse_args()
    
    if args.show_example:
        example = create_example_sample()
        print("Example sample format:")
        print(json.dumps(example, indent=2))
    else:
        prepare_vlm_samples_from_csv(
            csv_path=args.csv_path,
            classifier_predictions_json=args.predictions_json,
            output_json=args.output_json,
            heatmap_dir=args.heatmap_dir
        )
