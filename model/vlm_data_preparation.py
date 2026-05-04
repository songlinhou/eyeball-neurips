"""
Data preparation for VLM finetuning
Extracts important frames, generates heatmaps, and creates training data
"""

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
import json
from pathlib import Path
from typing import Dict, List, Tuple
import os


class VLMDataPreparator:
    """
    Prepares data for VLM finetuning by:
    1. Running multi-class classification
    2. Extracting important frames with attention maps
    3. Generating heatmap overlays
    4. Creating text prompts with predictions
    """
    
    def __init__(self, model, device='cuda', top_k_frames=5):
        """
        Args:
            model: MultiClassExplainableResNet3D model
            device: Device to run inference on
            top_k_frames: Number of top important frames to extract
        """
        self.model = model
        self.device = device
        self.top_k_frames = top_k_frames
        self.model.eval()
        
        # Class label mappings based on ERDES dataset structure
        # From metadata.csv
        self.diagnostic_labels = {
            0: "non_rd",  # Non-Retinal Detachment
            1: "rd"       # Retinal Detachment
        }
        
        self.subtype_labels = {
            0: "normal",           # Normal eye
            1: "macula_intact",    # RD with macula attached
            2: "macula_detached",  # RD with macula detached
            3: "pvd"               # Posterior Vitreous Detachment
        }
    
    def predict_video(self, video_tensor: torch.Tensor) -> Dict:
        """
        Run multi-class prediction on video
        
        Args:
            video_tensor: Video tensor (B, C, T, H, W)
            
        Returns:
            predictions: Dict with class predictions and probabilities
        """
        with torch.no_grad():
            video_tensor = video_tensor.to(self.device)
            outputs, attention = self.model(video_tensor, return_attention=True)
            
            # Get predictions
            diagnostic_probs = torch.softmax(outputs['diagnostic'], dim=1)
            subtype_probs = torch.softmax(outputs['subtype'], dim=1)
            
            diagnostic_pred = torch.argmax(diagnostic_probs, dim=1)
            subtype_pred = torch.argmax(subtype_probs, dim=1)
            
            predictions = {
                'diagnostic': {
                    'class_id': diagnostic_pred.cpu().item(),
                    'class_name': self.diagnostic_labels[diagnostic_pred.cpu().item()],
                    'confidence': diagnostic_probs.max(dim=1)[0].cpu().item(),
                    'probabilities': diagnostic_probs.cpu().numpy()
                },
                'subtype': {
                    'class_id': subtype_pred.cpu().item(),
                    'class_name': self.subtype_labels[subtype_pred.cpu().item()],
                    'confidence': subtype_probs.max(dim=1)[0].cpu().item(),
                    'probabilities': subtype_probs.cpu().numpy()
                },
                'attention': attention
            }
            
        return predictions
    
    def extract_important_frames_with_attention(self, video_tensor: torch.Tensor) -> Tuple:
        """
        Extract important frames and their attention maps
        
        Args:
            video_tensor: Video tensor (B, C, T, H, W)
            
        Returns:
            important_frames: (B, top_k, C, H, W)
            frame_indices: (B, top_k)
            importance_scores: (B, top_k)
            spatial_attention: (B, top_k, 1, H, W)
        """
        video_tensor = video_tensor.to(self.device)
        return self.model.extract_important_frames(video_tensor, top_k=self.top_k_frames)
    
    def generate_heatmap_overlay(self, frame: np.ndarray, attention_map: np.ndarray, 
                                 alpha=0.5, colormap='jet') -> np.ndarray:
        """
        Generate heatmap overlay on frame
        
        Args:
            frame: Original frame (H, W, 3) in RGB, values [0, 255]
            attention_map: Attention map (H_attn, W_attn), values [0, 1]
            alpha: Transparency of heatmap
            colormap: Matplotlib colormap name
            
        Returns:
            overlay: Frame with heatmap overlay (H, W, 3)
        """
        # Ensure frame is uint8
        if frame.max() <= 1.0:
            frame = (frame * 255).astype(np.uint8)
        else:
            frame = frame.astype(np.uint8)
        
        # Get frame dimensions
        H, W = frame.shape[:2]
        
        # Ensure attention map is 2D
        if len(attention_map.shape) > 2:
            attention_map = attention_map.squeeze()
        
        # Resize attention map to match frame size if needed
        if attention_map.shape[:2] != (H, W):
            attention_map = cv2.resize(attention_map, (W, H), interpolation=cv2.INTER_LINEAR)
        
        # Ensure attention map is exactly (H, W) after resize
        if len(attention_map.shape) > 2:
            attention_map = attention_map[:, :, 0] if attention_map.shape[2] == 1 else attention_map.mean(axis=2)
        
        # Normalize attention map
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
        
        # Apply colormap
        cmap = cm.get_cmap(colormap)
        heatmap = cmap(attention_map)[:, :, :3]  # Remove alpha channel
        heatmap = (heatmap * 255).astype(np.uint8)
        
        # Ensure frame has 3 channels
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 4:
            frame = frame[:, :, :3]
        
        # Ensure both have same shape before blending
        assert frame.shape == heatmap.shape, f"Shape mismatch: frame {frame.shape} vs heatmap {heatmap.shape}"
        
        # Blend
        overlay = cv2.addWeighted(frame, 1 - alpha, heatmap, alpha, 0)
        
        return overlay
    
    def create_prompt_with_predictions(self, predictions: Dict, 
                                      include_confidence=True) -> str:
        """
        Create text prompt with model predictions
        
        Args:
            predictions: Prediction dictionary from predict_video
            include_confidence: Whether to include confidence scores
            
        Returns:
            prompt: Text prompt string
        """
        diagnostic = predictions['diagnostic']
        subtype = predictions['subtype']
        
        if include_confidence:
            prompt = f"""The AI model has analyzed this ocular ultrasound video and made the following predictions:

Primary Diagnosis: {diagnostic['class_name']} (confidence: {diagnostic['confidence']:.2%})
Subtype Classification: {subtype['class_name']} (confidence: {subtype['confidence']:.2%})

Based on the highlighted regions in the images (shown as heatmaps), please explain:
1. Why this diagnosis is likely based on the visual features you observe
2. What specific anatomical structures or patterns support this classification
3. How the motion patterns (if visible) contribute to the diagnosis
4. Any potential differential diagnoses to consider

Please provide a detailed clinical reasoning for these predictions."""
        else:
            prompt = f"""The AI model predicts:
- Primary Diagnosis: {diagnostic['class_name']}
- Subtype: {subtype['class_name']}

Explain the clinical reasoning based on the highlighted features in the images."""
        
        return prompt
    
    def prepare_vlm_sample(self, video_tensor: torch.Tensor, 
                          video_id: str,
                          output_dir: str,
                          ground_truth: Dict = None) -> Dict:
        """
        Prepare complete VLM training sample
        
        Args:
            video_tensor: Video tensor (1, C, T, H, W)
            video_id: Unique identifier for the video
            output_dir: Directory to save frames and heatmaps
            ground_truth: Optional ground truth labels
            
        Returns:
            sample: Dictionary with all data for VLM training
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get predictions
        predictions = self.predict_video(video_tensor)
        
        # Extract important frames
        important_frames, frame_indices, importance_scores, spatial_attention = \
            self.extract_important_frames_with_attention(video_tensor)
        
        # Debug: Check shapes
        if important_frames.shape[0] == 0:
            raise ValueError(f"No frames extracted for video {video_id}")
        
        # Convert to numpy - handle batch dimension safely
        try:
            important_frames = important_frames.cpu().numpy()[0]  # (top_k, C, H, W)
            frame_indices = frame_indices.cpu().numpy()[0]  # (top_k,)
            importance_scores = importance_scores.cpu().numpy()[0]  # (top_k,)
            spatial_attention = spatial_attention.cpu().numpy()[0]  # (top_k, 1, H, W)
        except IndexError as e:
            raise IndexError(f"Shape mismatch for video {video_id}: "
                           f"important_frames={important_frames.shape}, "
                           f"frame_indices={frame_indices.shape}, "
                           f"importance_scores={importance_scores.shape}, "
                           f"spatial_attention={spatial_attention.shape}") from e
        
        # Save frames and heatmaps
        frame_paths = []
        heatmap_paths = []
        
        # Use actual number of frames extracted (may be less than top_k_frames)
        num_frames = len(important_frames)
        
        for k in range(num_frames):
            # Original frame (C, H, W) -> (H, W, C)
            frame = important_frames[k].transpose(1, 2, 0)
            
            # Denormalize if needed (assuming ImageNet normalization)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            frame = frame * std + mean
            frame = np.clip(frame, 0, 1)
            frame_rgb = (frame * 255).astype(np.uint8)
            
            # Attention map (1, H, W) -> (H, W)
            attention = spatial_attention[k, 0]
            
            # Generate heatmap overlay
            heatmap_overlay = self.generate_heatmap_overlay(frame_rgb, attention)
            
            # Save files
            frame_path = output_dir / f"{video_id}_frame_{k}_idx{frame_indices[k]}.jpg"
            heatmap_path = output_dir / f"{video_id}_heatmap_{k}_idx{frame_indices[k]}.jpg"
            
            cv2.imwrite(str(frame_path), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(heatmap_path), cv2.cvtColor(heatmap_overlay, cv2.COLOR_RGB2BGR))
            
            frame_paths.append(str(frame_path))
            heatmap_paths.append(str(heatmap_path))
        
        # Create prompt
        prompt = self.create_prompt_with_predictions(predictions)
        
        # Prepare sample
        sample = {
            'video_id': video_id,
            'predictions': {
                'diagnostic': predictions['diagnostic']['class_name'],
                'diagnostic_confidence': predictions['diagnostic']['confidence'],
                'subtype': predictions['subtype']['class_name'],
                'subtype_confidence': predictions['subtype']['confidence']
            },
            'frame_indices': frame_indices.tolist(),
            'importance_scores': importance_scores.tolist(),
            'frame_paths': frame_paths,
            'heatmap_paths': heatmap_paths,
            'prompt': prompt,
            'ground_truth': ground_truth,
            # Add summary and diagnosis_text at top level for easier access
            'summary': ground_truth.get('summary', '') if ground_truth else '',
            'diagnosis_text': ground_truth.get('diagnosis_text', '') if ground_truth else ''
        }
        
        # Save metadata
        metadata_path = output_dir / f"{video_id}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(sample, f, indent=2)
        
        return sample
    
    def create_contrastive_samples(self, video_tensor: torch.Tensor,
                                   video_id: str,
                                   output_dir: str,
                                   ground_truth: Dict = None) -> Tuple[Dict, Dict]:
        """
        Create contrastive samples: one with correct heatmaps, one with spatially shifted heatmaps
        This helps ensure the VLM actually uses the heatmap information through FAVG paradigm
        
        Args:
            video_tensor: Video tensor (1, C, T, H, W)
            video_id: Unique identifier
            output_dir: Output directory
            ground_truth: Ground truth labels (optional)
            
        Returns:
            correct_sample: Sample with correct heatmaps
            contrastive_sample: Sample with spatially shifted heatmaps (20-50% shift)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get correct sample
        correct_sample = self.prepare_vlm_sample(video_tensor, f"{video_id}_correct", 
                                                 output_dir / "correct", ground_truth)
        
        # Create contrastive sample with spatially shifted attention
        predictions = self.predict_video(video_tensor)
        important_frames, frame_indices, importance_scores, _ = \
            self.extract_important_frames_with_attention(video_tensor)
        
        # Convert to numpy - handle batch dimension safely
        try:
            important_frames = important_frames.cpu().numpy()[0]
            frame_indices = frame_indices.cpu().numpy()[0]
            importance_scores = importance_scores.cpu().numpy()[0]
        except IndexError as e:
            raise IndexError(f"Shape mismatch in contrastive samples for video {video_id}: "
                           f"important_frames={important_frames.shape}, "
                           f"frame_indices={frame_indices.shape}, "
                           f"importance_scores={importance_scores.shape}") from e
        
        # Generate spatially shifted attention maps for contrastive samples
        contrastive_paths = []
        
        # Use actual number of frames extracted (may be less than top_k_frames)
        num_frames = len(important_frames)
        
        # Get the actual attention maps for spatial shifting
        _, _, _, attention_maps = self.extract_important_frames_with_attention(video_tensor)
        attention_maps = attention_maps.cpu().numpy()[0]  # (num_frames, H, W)
        
        for k in range(num_frames):
            frame = important_frames[k].transpose(1, 2, 0)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            frame = frame * std + mean
            frame = np.clip(frame, 0, 1)
            frame_rgb = (frame * 255).astype(np.uint8)
            
            # Spatially shift the attention map instead of using random noise
            original_attention = attention_maps[k]
            att_H, att_W = original_attention.shape[:2]
            
            # Apply random spatial shift (between 20-50% of attention map dimensions)
            # Ensure minimum shift of 1 pixel
            min_shift_x = max(1, int(att_W * 0.2))
            max_shift_x = max(min_shift_x + 1, int(att_W * 0.5) + 1)
            min_shift_y = max(1, int(att_H * 0.2))
            max_shift_y = max(min_shift_y + 1, int(att_H * 0.5) + 1)
            
            shift_x = np.random.randint(min_shift_x, max_shift_x) * np.random.choice([-1, 1])
            shift_y = np.random.randint(min_shift_y, max_shift_y) * np.random.choice([-1, 1])
            
            # Shift the attention map using np.roll
            shifted_attention = np.roll(original_attention, shift=(shift_y, shift_x), axis=(0, 1))
            
            # Generate heatmap with shifted attention
            heatmap_overlay = self.generate_heatmap_overlay(frame_rgb, shifted_attention)
            
            heatmap_path = output_dir / "contrastive" / f"{video_id}_shifted_heatmap_{k}.jpg"
            heatmap_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(heatmap_path), cv2.cvtColor(heatmap_overlay, cv2.COLOR_RGB2BGR))
            contrastive_paths.append(str(heatmap_path))
        
        contrastive_sample = {
            'video_id': f"{video_id}_contrastive",
            'predictions': correct_sample['predictions'],
            'frame_indices': frame_indices.tolist(),
            'importance_scores': importance_scores.tolist(),
            'heatmap_paths': contrastive_paths,
            'prompt': correct_sample['prompt'],
            'is_contrastive': True,
            'note': 'This sample has spatially shifted heatmaps for contrastive learning'
        }
        
        return correct_sample, contrastive_sample


def batch_prepare_vlm_data(model, video_loader, output_dir, device='cuda', 
                           use_contrastive=True):
    """
    Batch process videos to prepare VLM training data
    
    Args:
        model: MultiClassExplainableResNet3D model
        video_loader: DataLoader with videos
        output_dir: Output directory
        device: Device to use
        use_contrastive: Whether to create contrastive samples
        
    Returns:
        all_samples: List of all prepared samples
    """
    preparator = VLMDataPreparator(model, device=device)
    all_samples = []
    
    for batch_idx, (videos, labels, video_ids) in enumerate(video_loader):
        for i in range(videos.shape[0]):
            video = videos[i:i+1]  # Keep batch dimension
            video_id = video_ids[i] if isinstance(video_ids, list) else f"video_{batch_idx}_{i}"
            
            if use_contrastive:
                correct, contrastive = preparator.create_contrastive_samples(
                    video, video_id, output_dir
                )
                all_samples.append(correct)
                all_samples.append(contrastive)
            else:
                sample = preparator.prepare_vlm_sample(
                    video, video_id, output_dir
                )
                all_samples.append(sample)
    
    # Save all samples metadata
    output_path = Path(output_dir) / "all_samples.json"
    with open(output_path, 'w') as f:
        json.dump(all_samples, f, indent=2)
    
    print(f"Prepared {len(all_samples)} samples")
    print(f"Saved to {output_dir}")
    
    return all_samples
