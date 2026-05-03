"""
Example script demonstrating the complete VLM diagnosis pipeline
"""

import torch
import numpy as np
from pathlib import Path

# Import our modules
from multiclass_model import create_multiclass_model
from vlm_data_preparation import VLMDataPreparator
from vlm_finetuning import setup_qwen2vl_for_finetuning, inference_vlm


def example_1_prepare_data():
    """
    Example 1: Prepare VLM training data from a video
    """
    print("="*80)
    print("EXAMPLE 1: Preparing VLM Training Data")
    print("="*80)
    
    # Load trained classifier
    print("\n1. Loading classifier...")
    model = create_multiclass_model(
        num_diagnostic_classes=3,
        num_subtype_classes=2,
        num_anatomical_classes=4,
        pretrained=True,
        dropout=0.3
    )
    
    # Load checkpoint (replace with your actual checkpoint)
    # model.load_state_dict(torch.load('classifier.pth'))
    model.eval()
    
    # Create dummy video for demonstration
    print("\n2. Creating dummy video...")
    video = torch.randn(1, 3, 32, 224, 224)  # (B, C, T, H, W)
    
    # Initialize data preparator
    print("\n3. Initializing VLM data preparator...")
    preparator = VLMDataPreparator(
        model=model,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        top_k_frames=5
    )
    
    # Prepare single sample
    print("\n4. Preparing VLM sample...")
    sample = preparator.prepare_vlm_sample(
        video_tensor=video,
        video_id="example_video_001",
        output_dir="./example_vlm_data"
    )
    
    print("\n5. Sample prepared!")
    print(f"   - Video ID: {sample['video_id']}")
    print(f"   - Predictions:")
    print(f"     * Diagnostic: {sample['predictions']['diagnostic']} "
          f"({sample['predictions']['diagnostic_confidence']:.1%})")
    print(f"     * Subtype: {sample['predictions']['subtype']} "
          f"({sample['predictions']['subtype_confidence']:.1%})")
    print(f"     * Anatomical: {sample['predictions']['anatomical']} "
          f"({sample['predictions']['anatomical_confidence']:.1%})")
    print(f"   - Important frames: {sample['frame_indices']}")
    print(f"   - Frame paths: {len(sample['frame_paths'])} files")
    print(f"   - Heatmap paths: {len(sample['heatmap_paths'])} files")
    
    print("\n6. Creating contrastive samples...")
    correct_sample, contrastive_sample = preparator.create_contrastive_samples(
        video_tensor=video,
        video_id="example_video_001",
        output_dir="./example_vlm_data"
    )
    
    print(f"   - Correct sample: {correct_sample['video_id']}")
    print(f"   - Contrastive sample: {contrastive_sample['video_id']}")
    print(f"   - Contrastive has random heatmaps: {contrastive_sample.get('is_contrastive', False)}")
    
    return sample


def example_2_extract_frames():
    """
    Example 2: Extract important frames and attention maps
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Extracting Important Frames")
    print("="*80)
    
    # Load model
    print("\n1. Loading classifier...")
    model = create_multiclass_model(
        num_diagnostic_classes=3,
        num_subtype_classes=2,
        num_anatomical_classes=4,
        pretrained=True
    )
    model.eval()
    
    # Create dummy video
    video = torch.randn(1, 3, 32, 224, 224)
    
    # Extract important frames
    print("\n2. Extracting important frames...")
    important_frames, frame_indices, importance_scores, spatial_attention = \
        model.extract_important_frames(video, top_k=5)
    
    print(f"\n3. Extraction results:")
    print(f"   - Important frames shape: {important_frames.shape}")  # (1, 5, 3, 224, 224)
    print(f"   - Frame indices: {frame_indices[0].tolist()}")
    print(f"   - Importance scores: {importance_scores[0].tolist()}")
    print(f"   - Spatial attention shape: {spatial_attention.shape}")  # (1, 5, 1, 224, 224)
    
    # Visualize frame importance
    print("\n4. Frame importance distribution:")
    for i, (idx, score) in enumerate(zip(frame_indices[0], importance_scores[0])):
        bar = "█" * int(score * 50)
        print(f"   Frame {i+1} (idx {idx:2d}): {bar} {score:.3f}")
    
    return important_frames, frame_indices, importance_scores, spatial_attention


def example_3_multi_class_prediction():
    """
    Example 3: Multi-class prediction with attention
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Multi-Class Prediction")
    print("="*80)
    
    # Load model
    print("\n1. Loading classifier...")
    model = create_multiclass_model(
        num_diagnostic_classes=3,
        num_subtype_classes=2,
        num_anatomical_classes=4,
        pretrained=True
    )
    model.eval()
    
    # Create dummy video
    video = torch.randn(1, 3, 32, 224, 224)
    
    # Run prediction
    print("\n2. Running multi-class prediction...")
    with torch.no_grad():
        outputs, attention = model(video, return_attention=True)
    
    # Get class predictions
    diagnostic_probs = torch.softmax(outputs['diagnostic'], dim=1)
    subtype_probs = torch.softmax(outputs['subtype'], dim=1)
    anatomical_probs = torch.softmax(outputs['anatomical'], dim=1)
    
    print("\n3. Predictions:")
    print(f"\n   Diagnostic Class:")
    for i, prob in enumerate(diagnostic_probs[0]):
        print(f"     Class {i}: {prob:.2%}")
    
    print(f"\n   Subtype:")
    for i, prob in enumerate(subtype_probs[0]):
        print(f"     Class {i}: {prob:.2%}")
    
    print(f"\n   Anatomical Location:")
    for i, prob in enumerate(anatomical_probs[0]):
        print(f"     Class {i}: {prob:.2%}")
    
    print(f"\n4. Attention maps:")
    print(f"   - Frame importance shape: {attention['frame_importance'].shape}")
    print(f"   - Spatial attention shape: {attention['spatial_attention'].shape}")
    
    return outputs, attention


def example_4_vlm_inference():
    """
    Example 4: VLM inference (requires finetuned model)
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: VLM Inference")
    print("="*80)
    
    print("\n1. This example requires a finetuned VLM model.")
    print("   To run this example:")
    print("   a) First prepare training data (Example 1)")
    print("   b) Finetune Qwen 2.5 VL using vlm_finetuning.py")
    print("   c) Then run inference with the finetuned model")
    
    print("\n2. Example code:")
    print("""
    from vlm_finetuning import setup_qwen2vl_for_finetuning, inference_vlm
    
    # Load finetuned model
    model, processor = setup_qwen2vl_for_finetuning(
        model_name="./vlm_finetuned/final_model"
    )
    
    # Prepare images and prompt
    image_paths = [
        "./example_vlm_data/example_video_001_heatmap_0.jpg",
        "./example_vlm_data/example_video_001_heatmap_1.jpg",
        # ... more frames
    ]
    
    prompt = \"\"\"The AI model predicts:
    - Diagnostic: Retinal Detachment (95%)
    - Subtype: Macula Detached (87%)
    - Location: Superior (92%)
    
    Based on the highlighted regions, explain the clinical reasoning...\"\"\"
    
    # Run inference
    reasoning = inference_vlm(
        model=model,
        processor=processor,
        image_paths=image_paths,
        prompt=prompt
    )
    
    print(reasoning)
    """)


def example_5_complete_pipeline():
    """
    Example 5: Complete diagnosis pipeline
    """
    print("\n" + "="*80)
    print("EXAMPLE 5: Complete Diagnosis Pipeline")
    print("="*80)
    
    print("\n1. Loading classifier...")
    model = create_multiclass_model(
        num_diagnostic_classes=3,
        num_subtype_classes=2,
        num_anatomical_classes=4,
        pretrained=True
    )
    model.eval()
    
    print("\n2. Creating test video...")
    video = torch.randn(1, 3, 32, 224, 224)
    
    print("\n3. Running classification...")
    with torch.no_grad():
        outputs, attention = model(video, return_attention=True)
    
    # Get predictions
    diagnostic_pred = torch.argmax(outputs['diagnostic'], dim=1).item()
    subtype_pred = torch.argmax(outputs['subtype'], dim=1).item()
    anatomical_pred = torch.argmax(outputs['anatomical'], dim=1).item()
    
    diagnostic_conf = torch.softmax(outputs['diagnostic'], dim=1).max().item()
    subtype_conf = torch.softmax(outputs['subtype'], dim=1).max().item()
    anatomical_conf = torch.softmax(outputs['anatomical'], dim=1).max().item()
    
    print(f"\n4. Classification Results:")
    print(f"   - Diagnostic: Class {diagnostic_pred} ({diagnostic_conf:.1%})")
    print(f"   - Subtype: Class {subtype_pred} ({subtype_conf:.1%})")
    print(f"   - Anatomical: Class {anatomical_pred} ({anatomical_conf:.1%})")
    
    print("\n5. Extracting important frames...")
    important_frames, frame_indices, importance_scores, spatial_attention = \
        model.extract_important_frames(video, top_k=5)
    
    print(f"   - Extracted {important_frames.shape[1]} frames")
    print(f"   - Frame indices: {frame_indices[0].tolist()}")
    
    print("\n6. Preparing VLM data...")
    preparator = VLMDataPreparator(model, device='cpu')
    sample = preparator.prepare_vlm_sample(
        video_tensor=video,
        video_id="complete_example",
        output_dir="./example_complete"
    )
    
    print(f"   - Saved {len(sample['frame_paths'])} frames")
    print(f"   - Saved {len(sample['heatmap_paths'])} heatmaps")
    print(f"   - Generated prompt ({len(sample['prompt'])} chars)")
    
    print("\n7. Complete pipeline finished!")
    print(f"   - Output directory: ./example_complete")
    print(f"   - Ready for VLM finetuning")
    
    return sample


def main():
    """Run all examples"""
    print("\n" + "="*80)
    print("VLM MEDICAL VIDEO DIAGNOSIS - EXAMPLES")
    print("="*80)
    
    print("\nThese examples demonstrate the complete pipeline:")
    print("1. Prepare VLM training data")
    print("2. Extract important frames")
    print("3. Multi-class prediction")
    print("4. VLM inference (requires finetuned model)")
    print("5. Complete diagnosis pipeline")
    
    # Run examples
    try:
        example_1_prepare_data()
        example_2_extract_frames()
        example_3_multi_class_prediction()
        example_4_vlm_inference()
        example_5_complete_pipeline()
        
        print("\n" + "="*80)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        print("\nNext steps:")
        print("1. Train the multi-class classifier on your dataset")
        print("2. Prepare VLM training data using your trained classifier")
        print("3. Finetune Qwen 2.5 VL with the prepared data")
        print("4. Run diagnosis on new videos")
        
        print("\nFor more details, see VLM_README.md")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        print("This is expected if you don't have a trained classifier yet.")
        print("The examples show the workflow - adapt them to your needs!")


if __name__ == "__main__":
    main()
