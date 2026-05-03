"""
Complete pipeline for VLM-based medical video diagnosis
Integrates ExplainableOpticalFlowResNet3D with Qwen 2.5 VL
"""

import torch
import argparse
from pathlib import Path
import json
from torch.utils.data import DataLoader

from multiclass_model import create_multiclass_model
from vlm_data_preparation import VLMDataPreparator, batch_prepare_vlm_data
from vlm_finetuning import setup_qwen2vl_for_finetuning, train_vlm, MedicalVLMDataset, inference_vlm


class VLMDiagnosisPipeline:
    """
    End-to-end pipeline for VLM-based medical video diagnosis
    
    Steps:
    1. Load pretrained ExplainableOpticalFlowResNet3D
    2. Extract important frames and attention maps
    3. Prepare VLM training data with heatmaps
    4. Finetune Qwen 2.5 VL with contrastive learning
    5. Inference with clinical reasoning
    """
    
    def __init__(self,
                 classifier_checkpoint: str,
                 num_diagnostic_classes: int = 2,
                 num_subtype_classes: int = 4,
                 device: str = 'cuda'):
        """
        Args:
            classifier_checkpoint: Path to trained classifier checkpoint
            num_diagnostic_classes: Number of diagnostic classes
            num_subtype_classes: Number of subtype classes
            device: Device to use
        """
        self.device = device
        
        # Load classifier model
        print("Loading ExplainableOpticalFlowResNet3D model...")
        self.classifier = create_multiclass_model(
            num_diagnostic_classes=num_diagnostic_classes,
            num_subtype_classes=num_subtype_classes,
            pretrained=False
        )
        
        # Load checkpoint
        checkpoint = torch.load(classifier_checkpoint, map_location=device)
        self.classifier.load_state_dict(checkpoint)
        self.classifier = self.classifier.to(device)
        self.classifier.eval()
        
        print("Classifier loaded successfully!")
        
        self.vlm_model = None
        self.vlm_processor = None
    
    def prepare_training_data(self,
                             video_loader: DataLoader,
                             output_dir: str,
                             use_contrastive: bool = True):
        """
        Step 1: Prepare VLM training data from videos
        
        Args:
            video_loader: DataLoader with videos
            output_dir: Output directory for prepared data
            use_contrastive: Whether to create contrastive samples
            
        Returns:
            samples: List of prepared samples
        """
        print("\n" + "="*80)
        print("STEP 1: Preparing VLM Training Data")
        print("="*80)
        
        samples = batch_prepare_vlm_data(
            model=self.classifier,
            video_loader=video_loader,
            output_dir=output_dir,
            device=self.device,
            use_contrastive=use_contrastive
        )
        
        print(f"\nPrepared {len(samples)} samples")
        print(f"Data saved to: {output_dir}")
        
        return samples
    
    def setup_vlm(self,
                  model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
                  use_lora: bool = True,
                  load_in_4bit: bool = True):
        """
        Step 2: Setup Qwen 2.5 VL for finetuning
        
        Args:
            model_name: Hugging Face model name
            use_lora: Whether to use LoRA
            load_in_4bit: Whether to load in 4-bit
        """
        print("\n" + "="*80)
        print("STEP 2: Setting up Qwen 2.5 VL")
        print("="*80)
        
        self.vlm_model, self.vlm_processor = setup_qwen2vl_for_finetuning(
            model_name=model_name,
            use_lora=use_lora,
            load_in_4bit=load_in_4bit
        )
        
        print("VLM setup complete!")
    
    def finetune_vlm(self,
                    samples_json: str,
                    output_dir: str,
                    num_epochs: int = 3,
                    batch_size: int = 2,
                    learning_rate: float = 2e-5,
                    val_split: float = 0.1):
        """
        Step 3: Finetune VLM with prepared data
        
        Args:
            samples_json: Path to prepared samples JSON
            output_dir: Output directory for finetuned model
            num_epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            val_split: Validation split ratio
        """
        print("\n" + "="*80)
        print("STEP 3: Finetuning Qwen 2.5 VL")
        print("="*80)
        
        if self.vlm_model is None:
            raise ValueError("VLM not setup. Call setup_vlm() first.")
        
        # Create datasets
        full_dataset = MedicalVLMDataset(samples_json, self.vlm_processor)
        
        # Split into train/val
        dataset_size = len(full_dataset)
        val_size = int(dataset_size * val_split)
        train_size = dataset_size - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
        
        print(f"Training samples: {train_size}")
        print(f"Validation samples: {val_size}")
        
        # Train
        trainer = train_vlm(
            model=self.vlm_model,
            processor=self.vlm_processor,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            output_dir=output_dir,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        
        print(f"\nFinetuning complete! Model saved to: {output_dir}")
        
        return trainer
    
    def diagnose_video(self,
                      video_tensor: torch.Tensor,
                      video_id: str = "test_video",
                      temp_dir: str = "./temp_diagnosis"):
        """
        Step 4: Run complete diagnosis on a video
        
        Args:
            video_tensor: Video tensor (1, C, T, H, W)
            video_id: Video identifier
            temp_dir: Temporary directory for intermediate files
            
        Returns:
            diagnosis: Complete diagnosis with predictions and reasoning
        """
        print("\n" + "="*80)
        print(f"DIAGNOSING VIDEO: {video_id}")
        print("="*80)
        
        if self.vlm_model is None:
            raise ValueError("VLM not setup. Call setup_vlm() first.")
        
        # Prepare data for this video
        preparator = VLMDataPreparator(self.classifier, device=self.device)
        sample = preparator.prepare_vlm_sample(
            video_tensor=video_tensor,
            video_id=video_id,
            output_dir=temp_dir
        )
        
        print("\n1. Classifier Predictions:")
        print(f"   - Diagnostic: {sample['predictions']['diagnostic']} "
              f"({sample['predictions']['diagnostic_confidence']:.1%})")
        print(f"   - Subtype: {sample['predictions']['subtype']} "
              f"({sample['predictions']['subtype_confidence']:.1%})")
        
        print("\n2. Important Frames:")
        for i, (idx, score) in enumerate(zip(sample['frame_indices'], sample['importance_scores'])):
            print(f"   Frame {i+1}: Index {idx}, Importance {score:.3f}")
        
        print("\n3. Generating Clinical Reasoning with VLM...")
        
        # Run VLM inference
        reasoning = inference_vlm(
            model=self.vlm_model,
            processor=self.vlm_processor,
            image_paths=sample['heatmap_paths'],
            prompt=sample['prompt'],
            device=self.device
        )
        
        diagnosis = {
            'video_id': video_id,
            'predictions': sample['predictions'],
            'important_frames': {
                'indices': sample['frame_indices'],
                'scores': sample['importance_scores'],
                'paths': sample['frame_paths'],
                'heatmap_paths': sample['heatmap_paths']
            },
            'clinical_reasoning': reasoning
        }
        
        print("\n4. Clinical Reasoning:")
        print("-" * 80)
        print(reasoning)
        print("-" * 80)
        
        # Save diagnosis
        diagnosis_path = Path(temp_dir) / f"{video_id}_diagnosis.json"
        with open(diagnosis_path, 'w') as f:
            json.dump(diagnosis, f, indent=2)
        
        print(f"\nDiagnosis saved to: {diagnosis_path}")
        
        return diagnosis


def main():
    parser = argparse.ArgumentParser(description='VLM Medical Video Diagnosis Pipeline')
    
    # Mode selection
    parser.add_argument('--mode', type=str, required=True,
                       choices=['prepare', 'finetune', 'diagnose', 'full'],
                       help='Pipeline mode')
    
    # Classifier arguments
    parser.add_argument('--classifier_checkpoint', type=str, required=True,
                       help='Path to classifier checkpoint')
    parser.add_argument('--num_diagnostic_classes', type=int, default=2)
    parser.add_argument('--num_subtype_classes', type=int, default=4)
    
    # Data preparation arguments
    parser.add_argument('--video_dir', type=str, help='Directory with videos')
    parser.add_argument('--data_output_dir', type=str, default='./vlm_data',
                       help='Output directory for prepared data')
    parser.add_argument('--use_contrastive', action='store_true',
                       help='Create contrastive samples')
    
    # VLM arguments
    parser.add_argument('--vlm_model', type=str, default='Qwen/Qwen2-VL-7B-Instruct',
                       help='VLM model name')
    parser.add_argument('--use_lora', action='store_true', help='Use LoRA')
    parser.add_argument('--load_in_4bit', action='store_true', help='Load in 4-bit')
    
    # Training arguments
    parser.add_argument('--samples_json', type=str, help='Path to samples JSON')
    parser.add_argument('--vlm_output_dir', type=str, default='./vlm_finetuned',
                       help='Output directory for finetuned VLM')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    
    # Inference arguments
    parser.add_argument('--test_video', type=str, help='Path to test video')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = VLMDiagnosisPipeline(
        classifier_checkpoint=args.classifier_checkpoint,
        num_diagnostic_classes=args.num_diagnostic_classes,
        num_subtype_classes=args.num_subtype_classes
    )
    
    if args.mode == 'prepare' or args.mode == 'full':
        # Prepare data
        # Note: You need to implement video loading logic
        print("Data preparation mode")
        print("Note: Implement video loading from", args.video_dir)
        
    if args.mode == 'finetune' or args.mode == 'full':
        # Setup and finetune VLM
        pipeline.setup_vlm(
            model_name=args.vlm_model,
            use_lora=args.use_lora,
            load_in_4bit=args.load_in_4bit
        )
        
        pipeline.finetune_vlm(
            samples_json=args.samples_json,
            output_dir=args.vlm_output_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
    
    if args.mode == 'diagnose':
        # Run diagnosis
        pipeline.setup_vlm(model_name=args.vlm_model)
        
        # Load test video
        # Note: Implement video loading
        print("Diagnosis mode")
        print("Note: Implement video loading from", args.test_video)


if __name__ == "__main__":
    main()
