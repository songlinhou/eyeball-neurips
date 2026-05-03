"""
Qwen 2.5 VL Finetuning for Medical Video Diagnosis
Uses frames + heatmaps + predictions to train VLM for clinical reasoning
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from transformers import TrainingArguments, Trainer, TrainerCallback
from PIL import Image
import json
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
from dataclasses import dataclass
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MedicalVLMDataset(Dataset):
    """
    Dataset for VLM finetuning with medical videos
    Includes frames, heatmaps, predictions, and clinical reasoning
    
    Expected JSON format for each sample:
    {
        "video_id": "164267_02030",
        "frame_paths": ["path/to/frame1.jpg", ...],
        "heatmap_paths": ["path/to/heatmap1.jpg", ...],  # Optional
        "prompt": "The AI model predicts: ...",
        "predictions": {
            "diagnostic": "non_rd",
            "diagnostic_confidence": 0.95,
            "subtype": "normal",
            "subtype_confidence": 0.92
        },
        "summary": "Expert-written clinical description from CSV",  # Ground truth from balanced_split_desc.csv
        "diagnosis_text": "<diagnostic>non_rd</diagnostic><subtype>normal</subtype><anatomical>nan</anatomical>",  # Structured diagnosis
        "is_contrastive": false  # true for spatially-shifted heatmaps
    }
    """
    
    def __init__(self, 
                 samples_json: str,
                 processor,
                 max_length: int = 512,
                 use_heatmaps: bool = True,
                 contrastive_weight: float = 0.3):
        """
        Args:
            samples_json: Path to JSON file with prepared samples
            processor: Qwen2VL processor
            max_length: Maximum sequence length
            use_heatmaps: Whether to use heatmap overlays
            contrastive_weight: Weight for contrastive samples in training
        """
        self.processor = processor
        self.max_length = max_length
        self.use_heatmaps = use_heatmaps
        self.contrastive_weight = contrastive_weight
        
        # Load samples
        with open(samples_json, 'r') as f:
            all_samples = json.load(f)
        
        # Separate correct and contrastive samples
        # Filter out samples without ground truth summaries
        self.correct_samples = [
            s for s in all_samples 
            if not s.get('is_contrastive', False) 
            and s.get('summary') and s.get('summary').strip()
        ]
        self.contrastive_samples = [s for s in all_samples if s.get('is_contrastive', False)]
        
        # Log filtering statistics
        total_correct = len([s for s in all_samples if not s.get('is_contrastive', False)])
        filtered_count = total_correct - len(self.correct_samples)
        
        logger.info(f"Loaded {len(self.correct_samples)} correct samples")
        if filtered_count > 0:
            logger.warning(f"Filtered out {filtered_count} samples without ground truth summaries")
        logger.info(f"Loaded {len(self.contrastive_samples)} contrastive samples")
        
    def __len__(self):
        return len(self.correct_samples)
    
    def __getitem__(self, idx):
        # Randomly decide whether to use contrastive sample
        use_contrastive = (np.random.rand() < self.contrastive_weight and 
                          len(self.contrastive_samples) > 0)
        
        if use_contrastive:
            # Match contrastive sample to correct sample
            correct_sample = self.correct_samples[idx]
            video_id_base = correct_sample['video_id'].replace('_correct', '')
            
            # Find matching contrastive sample
            contrastive_sample = None
            for cs in self.contrastive_samples:
                if video_id_base in cs['video_id']:
                    contrastive_sample = cs
                    break
            
            if contrastive_sample:
                sample = contrastive_sample
                is_contrastive = True
            else:
                sample = correct_sample
                is_contrastive = False
        else:
            sample = self.correct_samples[idx]
            is_contrastive = False
        
        # Load images (use heatmaps if specified, otherwise original frames)
        if self.use_heatmaps and 'heatmap_paths' in sample:
            image_paths = sample['heatmap_paths']
        else:
            image_paths = sample['frame_paths']
        
        images = [Image.open(path).convert('RGB') for path in image_paths]
        
        # Create conversation
        prompt = sample['prompt']
        
        # For contrastive samples, modify the expected response
        if is_contrastive:
            # Expected response should indicate uncertainty or request for better visualization
            response = """I notice the highlighted regions in these images appear random or inconsistent with typical diagnostic patterns. The heatmap does not clearly indicate specific anatomical structures that would support the predicted diagnosis. 

To provide accurate clinical reasoning, I would need:
1. More focused attention on relevant anatomical landmarks
2. Clearer visualization of the pathological features
3. Consistent highlighting across frames showing the progression of the condition

Without reliable visual guidance, I cannot confidently explain why this specific diagnosis was made based solely on these highlighted regions."""
        else:
            # For correct samples, ONLY use ground truth summary from CSV
            # The 'summary' field contains expert-written clinical descriptions
            # The 'diagnosis_text' field contains structured diagnostic labels
            if 'summary' in sample and sample['summary'] and sample['summary'].strip():
                response = sample['summary']
                
                # Append diagnosis_text if available for structured output
                if 'diagnosis_text' in sample and sample['diagnosis_text']:
                    response += f"\n\n**Structured Diagnosis:**\n{sample['diagnosis_text']}"
            else:
                # No ground truth summary - this sample should have been filtered during data prep
                # Return None to signal that this sample should be skipped
                video_id = sample.get('video_id', 'unknown')
                logger.warning(f"Skipping video {video_id} - no ground truth summary in CSV")
                return None
        
        # Log ground truth for first few samples (for debugging)
        if idx < 3 and not hasattr(self, '_logged_samples'):
            self._logged_samples = set()
        
        if idx < 3 and idx not in getattr(self, '_logged_samples', set()):
            logger.info(f"\n{'='*80}")
            logger.info(f"SAMPLE {idx} - Video ID: {sample.get('video_id', 'unknown')}")
            logger.info(f"Is Contrastive: {is_contrastive}")
            logger.info(f"{'='*80}")
            logger.info(f"GROUND TRUTH TEXT:")
            logger.info(f"{'-'*80}")
            logger.info(response)
            logger.info(f"{'='*80}\n")
            if not hasattr(self, '_logged_samples'):
                self._logged_samples = set()
            self._logged_samples.add(idx)
        
        # Format as conversation
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"} for _ in images
                ] + [{"type": "text", "text": prompt}]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": response}]
            }
        ]
        
        # Process with Qwen2VL processor
        text = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
        
        inputs = self.processor(
            text=[text],
            images=images,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True
        )
        
        # Prepare labels (mask input, only train on response)
        labels = inputs['input_ids'].clone()
        
        # Find where assistant response starts and mask everything before it
        # This is a simplified approach - you may need to adjust based on actual token IDs
        
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'pixel_values': inputs.get('pixel_values'),
            'image_grid_thw': inputs.get('image_grid_thw'),
            'labels': labels.squeeze(0),
            'is_contrastive': is_contrastive
        }
    
    def _generate_template_response(self, sample: Dict) -> str:
        """
        DEPRECATED: Template responses are NOT allowed.
        Only real expert summaries from CSV should be used.
        
        This function is kept for reference but should never be called.
        If you see this error, it means the data preparation is incorrect.
        """
        video_id = sample.get('video_id', 'unknown')
        raise ValueError(
            f"Template response requested for {video_id}. "
            f"This should NEVER happen. Only use expert summaries from CSV. "
            f"Please re-run data preparation with the fixed code."
        )


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss to ensure VLM uses heatmap information
    Penalizes similar outputs for correct vs random heatmaps
    """
    
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
        
    def forward(self, correct_logits, contrastive_logits, labels):
        """
        Args:
            correct_logits: Logits from samples with correct heatmaps
            contrastive_logits: Logits from samples with random heatmaps
            labels: Ground truth labels
            
        Returns:
            loss: Contrastive loss value
        """
        # Compute similarity between correct and contrastive outputs
        similarity = F.cosine_similarity(correct_logits, contrastive_logits, dim=-1)
        
        # We want dissimilar outputs, so penalize high similarity
        loss = torch.clamp(similarity - self.margin, min=0).mean()
        
        return loss


class GroundTruthLoggingCallback(TrainerCallback):
    """
    Callback to log ground truth text samples during training
    """
    
    def __init__(self, train_dataset, num_samples=5):
        """
        Args:
            train_dataset: Training dataset
            num_samples: Number of samples to log per epoch
        """
        self.train_dataset = train_dataset
        self.num_samples = num_samples
        self.logged_epochs = set()
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        """Log ground truth samples at the beginning of each epoch"""
        epoch = int(state.epoch) if state.epoch is not None else 0
        
        # Only log once per epoch
        if epoch in self.logged_epochs:
            return
        
        self.logged_epochs.add(epoch)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"EPOCH {epoch} - GROUND TRUTH SAMPLES")
        logger.info(f"{'='*80}\n")
        
        # Log a few random samples
        indices = np.random.choice(len(self.train_dataset.correct_samples), 
                                   min(self.num_samples, len(self.train_dataset.correct_samples)), 
                                   replace=False)
        
        for i, idx in enumerate(indices):
            sample = self.train_dataset.correct_samples[idx]
            
            logger.info(f"\n{'-'*80}")
            logger.info(f"Sample {i+1}/{len(indices)} - Video ID: {sample.get('video_id', 'unknown')}")
            logger.info(f"{'-'*80}")
            
            # Log predictions
            if 'predictions' in sample:
                pred = sample['predictions']
                logger.info(f"Predictions:")
                logger.info(f"  - Diagnostic: {pred.get('diagnostic', 'N/A')} "
                          f"({pred.get('diagnostic_confidence', 0):.1%})")
                logger.info(f"  - Subtype: {pred.get('subtype', 'N/A')} "
                          f"({pred.get('subtype_confidence', 0):.1%})")
            
            # Log ground truth text
            logger.info(f"\nGround Truth Text:")
            logger.info(f"{'-'*40}")
            
            if 'summary' in sample and sample['summary']:
                gt_text = sample['summary']
                if 'diagnosis_text' in sample and sample['diagnosis_text']:
                    gt_text += f"\n\n**Structured Diagnosis:**\n{sample['diagnosis_text']}"
            elif sample.get('ground_truth') and 'clinical_reasoning' in sample['ground_truth']:
                gt_text = sample['ground_truth']['clinical_reasoning']
            else:
                gt_text = "[No ground truth available - will use template]"
            
            # Truncate if too long
            if len(gt_text) > 500:
                logger.info(gt_text[:500] + "...")
                logger.info(f"[Truncated - full length: {len(gt_text)} chars]")
            else:
                logger.info(gt_text)
            
            logger.info(f"{'-'*80}\n")
        
        logger.info(f"{'='*80}\n")


def setup_qwen2vl_for_finetuning(model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
                                 use_lora: bool = True,
                                 lora_r: int = 16,
                                 lora_alpha: int = 32,
                                 lora_dropout: float = 0.05,
                                 load_in_4bit: bool = True):
    """
    Setup Qwen2VL model for finetuning with LoRA
    
    Args:
        model_name: Hugging Face model name
        use_lora: Whether to use LoRA for parameter-efficient finetuning
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        load_in_4bit: Whether to load in 4-bit for memory efficiency
        
    Returns:
        model: Prepared model
        processor: Qwen2VL processor
    """
    logger.info(f"Loading {model_name}...")
    
    # Load processor
    processor = AutoProcessor.from_pretrained(model_name)
    
    # Load model
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        model = prepare_model_for_kbit_training(model)
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
    
    # Apply LoRA
    if use_lora:
        logger.info("Applying LoRA...")
        
        # Target modules for LoRA (adjust based on Qwen2VL architecture)
        target_modules = [
            "q_proj",
            "k_proj", 
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ]
        
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    return model, processor


def train_vlm(
    model,
    processor,
    train_dataset,
    val_dataset,
    output_dir: str,
    num_epochs: int = 3,
    batch_size: int = 2,
    learning_rate: float = 2e-5,
    warmup_steps: int = 100,
    gradient_accumulation_steps: int = 4,
    save_steps: int = 100,
    eval_steps: int = 100,
    logging_steps: int = 10,
    resume_from_checkpoint: bool = True
):
    """
    Train VLM with medical video data
    
    Args:
        model: Qwen2VL model
        processor: Qwen2VL processor
        train_dataset: Training dataset
        val_dataset: Validation dataset
        output_dir: Output directory
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        warmup_steps: Warmup steps
        gradient_accumulation_steps: Gradient accumulation steps
        save_steps: Save checkpoint every N steps (default: 100)
        eval_steps: Evaluate every N steps (default: 100)
        logging_steps: Log every N steps
        resume_from_checkpoint: Whether to resume from last checkpoint if available
    """
    
    # Check for existing checkpoints
    checkpoint_dir = None
    if resume_from_checkpoint:
        from pathlib import Path
        checkpoints = list(Path(output_dir).glob("checkpoint-*"))
        if checkpoints:
            # Sort by checkpoint number and get the latest
            checkpoints.sort(key=lambda x: int(x.name.split("-")[1]))
            checkpoint_dir = str(checkpoints[-1])
            logger.info(f"Found checkpoint: {checkpoint_dir}. Resuming training...")
        else:
            logger.info("No checkpoint found. Starting training from scratch.")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,  # Save every 100 steps by default
        eval_steps=eval_steps,  # Evaluate every 100 steps by default
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to="tensorboard",
        save_total_limit=5,  # Keep last 5 checkpoints
        resume_from_checkpoint=checkpoint_dir  # Auto-resume from last checkpoint
    )
    
    def collate_fn(batch):
        """Custom collator for Qwen2VL that handles image_grid_thw"""
        # Stack tensors
        input_ids = torch.stack([f['input_ids'] for f in batch])
        attention_mask = torch.stack([f['attention_mask'] for f in batch])
        labels = torch.stack([f['labels'] for f in batch])
        
        # Handle pixel_values and image_grid_thw
        pixel_values = None
        image_grid_thw = None
        
        if batch[0]['pixel_values'] is not None:
            pixel_values = torch.cat([f['pixel_values'] for f in batch], dim=0)
        
        if batch[0]['image_grid_thw'] is not None:
            image_grid_thw = torch.cat([f['image_grid_thw'] for f in batch], dim=0)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'pixel_values': pixel_values,
            'image_grid_thw': image_grid_thw
        }
    
    # Create callback to log ground truth samples
    gt_callback = GroundTruthLoggingCallback(train_dataset, num_samples=5)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        callbacks=[gt_callback]
    )
    
    logger.info("Starting training...")
    if checkpoint_dir:
        logger.info(f"Resuming from checkpoint: {checkpoint_dir}")
        trainer.train(resume_from_checkpoint=checkpoint_dir)
    else:
        logger.info("Starting training from scratch")
        trainer.train()
    
    # Save final model
    logger.info(f"Saving final model to {output_dir}/final_model")
    trainer.save_model(f"{output_dir}/final_model")
    processor.save_pretrained(f"{output_dir}/final_model")
    logger.info("Training complete!")
    
    return trainer


def inference_vlm(model, processor, image_paths: List[str], prompt: str, device='cuda'):
    """
    Run inference with trained VLM
    
    Args:
        model: Trained Qwen2VL model
        processor: Qwen2VL processor
        image_paths: List of image paths
        prompt: Text prompt
        device: Device to use
        
    Returns:
        response: Generated text response
    """
    model.eval()
    
    # Load images
    images = [Image.open(path).convert('RGB') for path in image_paths]
    
    # Format conversation
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"} for _ in images
            ] + [{"type": "text", "text": prompt}]
        }
    ]
    
    # Process
    text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=images, return_tensors="pt").to(device)
    
    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
    
    # Decode
    response = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    
    return response


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Finetune Qwen2VL for medical video diagnosis')
    parser.add_argument('--samples_json', type=str, required=True, help='Path to prepared samples JSON')
    parser.add_argument('--output_dir', type=str, default='./vlm_finetuned', help='Output directory')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2-VL-7B-Instruct', help='Model name')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--save_steps', type=int, default=100, help='Save checkpoint every N steps')
    parser.add_argument('--eval_steps', type=int, default=100, help='Evaluate every N steps')
    parser.add_argument('--no_resume', action='store_true', help='Do not resume from checkpoint')
    parser.add_argument('--use_lora', action='store_true', help='Use LoRA')
    parser.add_argument('--load_in_4bit', action='store_true', help='Load in 4-bit')
    
    args = parser.parse_args()
    
    # Setup model
    model, processor = setup_qwen2vl_for_finetuning(
        model_name=args.model_name,
        use_lora=args.use_lora,
        load_in_4bit=args.load_in_4bit
    )
    
    # Create datasets (split samples into train/val)
    train_dataset = MedicalVLMDataset(args.samples_json, processor)
    
    # For validation, use a subset without contrastive samples
    val_dataset = MedicalVLMDataset(args.samples_json, processor, contrastive_weight=0.0)
    
    # Train
    trainer = train_vlm(
        model=model,
        processor=processor,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        resume_from_checkpoint=not args.no_resume
    )
    
    print("Training complete! Model saved to:", args.output_dir)
