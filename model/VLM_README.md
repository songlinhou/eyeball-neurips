## VLM-Based Medical Video Diagnosis System

Complete pipeline for finetuning Qwen 2.5 VL to provide clinical reasoning for ultrasound video diagnosis using ExplainableOpticalFlowResNet3D.

## Overview

This system combines:
1. **ExplainableOpticalFlowResNet3D**: Multi-class video classifier with attention mechanisms
2. **Frame Extraction**: Identifies most important diagnostic frames
3. **Heatmap Generation**: Creates visual attention overlays
4. **Qwen 2.5 VL**: Vision-language model for clinical reasoning
5. **Contrastive Learning**: Ensures VLM actually uses heatmap information

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Input: Medical Video                      │
│                      (B, 3, 32, 224, 224)                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│         ExplainableOpticalFlowResNet3D Classifier            │
│  ┌──────────────────┐        ┌──────────────────┐          │
│  │   RGB Stream     │        │   Flow Stream    │          │
│  │  + Frame Attn    │        │  (Motion)        │          │
│  │  + Spatial Attn  │        │                  │          │
│  └────────┬─────────┘        └────────┬─────────┘          │
│           └──────────┬────────────────┘                     │
│                      ▼                                       │
│           ┌──────────────────────┐                          │
│           │  Multi-Class Output  │                          │
│           │  - Diagnostic Class  │                          │
│           │  - Subtype          │                          │
│           │  - Anatomical       │                          │
│           └──────────┬───────────┘                          │
└──────────────────────┼──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Frame & Attention Extraction                    │
│  ┌────────────────────────────────────────────────┐         │
│  │  Top-K Important Frames (based on frame attn)  │         │
│  │  + Spatial Attention Maps for each frame       │         │
│  └────────────────────┬───────────────────────────┘         │
└───────────────────────┼─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                  Heatmap Generation                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Frame 1      │  │ Frame 2      │  │ Frame K      │      │
│  │ + Heatmap    │  │ + Heatmap    │  │ + Heatmap    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              Prompt Construction                             │
│  ┌────────────────────────────────────────────────┐         │
│  │  "The AI model predicts:                       │         │
│  │   - Diagnostic: Retinal Detachment (95%)       │         │
│  │   - Subtype: Macula Detached (87%)             │         │
│  │   - Location: Superior (92%)                   │         │
│  │                                                 │         │
│  │  Based on highlighted regions, explain why..." │         │
│  └────────────────────────────────────────────────┘         │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                  Qwen 2.5 VL Model                           │
│  ┌────────────────────────────────────────────────┐         │
│  │  Input: Frames + Heatmaps + Prompt             │         │
│  │         ↓                                       │         │
│  │  Vision Encoder → Language Model                │         │
│  │         ↓                                       │         │
│  │  Output: Clinical Reasoning Text               │         │
│  │  "The highlighted regions show..."             │         │
│  └────────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

## Files

### Core Components

1. **`multiclass_model.py`** (320 lines)
   - Multi-class ExplainableOpticalFlowResNet3D
   - Outputs: diagnostic_class, subtype, anatomical_subclass
   - Method: `extract_important_frames()` - extracts top-K frames with attention

2. **`vlm_data_preparation.py`** (450 lines)
   - `VLMDataPreparator` class
   - Runs classification and extracts frames
   - Generates heatmap overlays
   - Creates text prompts with predictions
   - **Contrastive sample generation**: Creates pairs with correct/random heatmaps

3. **`vlm_finetuning.py`** (500 lines)
   - `MedicalVLMDataset`: Dataset with contrastive learning
   - `setup_qwen2vl_for_finetuning()`: Model setup with LoRA
   - `train_vlm()`: Training loop
   - `inference_vlm()`: Inference function

4. **`vlm_pipeline.py`** (350 lines)
   - `VLMDiagnosisPipeline`: End-to-end pipeline
   - Integrates all components
   - CLI interface

## Installation

```bash
# Install dependencies
pip install torch torchvision transformers
pip install peft accelerate bitsandbytes
pip install opencv-python matplotlib pillow
pip install qwen-vl-utils  # For Qwen 2.5 VL

# Install from Hugging Face
pip install git+https://github.com/huggingface/transformers
```

## Usage

### Step 1: Train Multi-Class Classifier

First, train the ExplainableOpticalFlowResNet3D model:

```python
from multiclass_model import create_multiclass_model

# Create model
model = create_multiclass_model(
    num_diagnostic_classes=3,  # Normal, RD, PVD
    num_subtype_classes=2,     # Macula On/Off
    num_anatomical_classes=4,  # Superior, Inferior, Temporal, Total
    pretrained=True,
    dropout=0.3
)

# Train (use existing training script or implement custom)
# Save checkpoint: torch.save(model.state_dict(), 'classifier.pth')
```

### Step 2: Prepare VLM Training Data

Extract frames, generate heatmaps, and create training samples:

```python
from vlm_data_preparation import VLMDataPreparator, batch_prepare_vlm_data
from multiclass_model import create_multiclass_model
import torch

# Load trained classifier
model = create_multiclass_model(...)
model.load_state_dict(torch.load('classifier.pth'))
model.eval()

# Prepare data
preparator = VLMDataPreparator(model, device='cuda', top_k_frames=5)

# For single video
video_tensor = ...  # (1, 3, 32, 224, 224)
sample = preparator.prepare_vlm_sample(
    video_tensor=video_tensor,
    video_id="video_001",
    output_dir="./vlm_data"
)

# For batch processing
samples = batch_prepare_vlm_data(
    model=model,
    video_loader=your_dataloader,
    output_dir="./vlm_data",
    use_contrastive=True  # Creates contrastive samples
)
```

### Step 3: Finetune Qwen 2.5 VL

```python
from vlm_finetuning import setup_qwen2vl_for_finetuning, train_vlm, MedicalVLMDataset

# Setup model with LoRA
model, processor = setup_qwen2vl_for_finetuning(
    model_name="Qwen/Qwen2-VL-7B-Instruct",
    use_lora=True,
    load_in_4bit=True  # For memory efficiency
)

# Create dataset
train_dataset = MedicalVLMDataset(
    samples_json="./vlm_data/all_samples.json",
    processor=processor,
    use_heatmaps=True,
    contrastive_weight=0.3  # 30% contrastive samples
)

# Train
trainer = train_vlm(
    model=model,
    processor=processor,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    output_dir="./vlm_finetuned",
    num_epochs=3,
    batch_size=2,
    learning_rate=2e-5
)
```

### Step 4: Run Diagnosis

```python
from vlm_pipeline import VLMDiagnosisPipeline

# Initialize pipeline
pipeline = VLMDiagnosisPipeline(
    classifier_checkpoint='classifier.pth',
    num_diagnostic_classes=3,
    num_subtype_classes=2,
    num_anatomical_classes=4
)

# Setup VLM
pipeline.setup_vlm(model_name="./vlm_finetuned/final_model")

# Diagnose video
video_tensor = ...  # (1, 3, 32, 224, 224)
diagnosis = pipeline.diagnose_video(
    video_tensor=video_tensor,
    video_id="test_video_001"
)

print(diagnosis['clinical_reasoning'])
```

### Command Line Interface

```bash
# Prepare data
python vlm_pipeline.py \
    --mode prepare \
    --classifier_checkpoint classifier.pth \
    --video_dir ./videos \
    --data_output_dir ./vlm_data \
    --use_contrastive

# Finetune VLM
python vlm_pipeline.py \
    --mode finetune \
    --classifier_checkpoint classifier.pth \
    --samples_json ./vlm_data/all_samples.json \
    --vlm_output_dir ./vlm_finetuned \
    --use_lora \
    --load_in_4bit \
    --epochs 3 \
    --batch_size 2

# Run diagnosis
python vlm_pipeline.py \
    --mode diagnose \
    --classifier_checkpoint classifier.pth \
    --vlm_model ./vlm_finetuned/final_model \
    --test_video ./test_video.mp4
```

## Contrastive Learning Strategy

To ensure the VLM actually uses heatmap information:

### 1. Correct Samples
- Frames with **correct** spatial attention heatmaps
- Expected response: Detailed clinical reasoning based on highlighted regions

### 2. Contrastive Samples
- Same frames with **random** attention heatmaps
- Expected response: Uncertainty, request for better visualization

### 3. Training Objective
```python
# Dataset automatically mixes correct and contrastive samples
train_dataset = MedicalVLMDataset(
    samples_json="data.json",
    processor=processor,
    contrastive_weight=0.3  # 30% contrastive
)

# Model learns to:
# - Provide confident reasoning when heatmaps are meaningful
# - Express uncertainty when heatmaps are random
```

## Output Structure

```
vlm_data/
├── correct/
│   ├── video_001_correct/
│   │   ├── video_001_correct_frame_0_idx5.jpg
│   │   ├── video_001_correct_heatmap_0_idx5.jpg
│   │   ├── video_001_correct_frame_1_idx12.jpg
│   │   ├── video_001_correct_heatmap_1_idx12.jpg
│   │   └── video_001_correct_metadata.json
│   └── ...
├── contrastive/
│   ├── video_001_contrastive/
│   │   ├── video_001_random_heatmap_0.jpg
│   │   ├── video_001_random_heatmap_1.jpg
│   │   └── ...
│   └── ...
└── all_samples.json

vlm_finetuned/
├── checkpoint-500/
├── checkpoint-1000/
├── final_model/
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   ├── config.json
│   └── preprocessor_config.json
└── training_logs/
```

## Example Diagnosis Output

```json
{
  "video_id": "test_video_001",
  "predictions": {
    "diagnostic": "Retinal Detachment",
    "diagnostic_confidence": 0.95,
    "subtype": "Macula Detached",
    "subtype_confidence": 0.87,
    "anatomical": "Superior",
    "anatomical_confidence": 0.92
  },
  "important_frames": {
    "indices": [5, 12, 18, 24, 29],
    "scores": [0.24, 0.21, 0.19, 0.18, 0.18],
    "paths": ["frame_0.jpg", ...],
    "heatmap_paths": ["heatmap_0.jpg", ...]
  },
  "clinical_reasoning": "Based on the analysis of the highlighted regions in these ultrasound frames, here is my clinical reasoning:\n\n**Primary Diagnosis: Retinal Detachment**\n\n1. **Visual Features Supporting Diagnosis:**\nThe highlighted regions show characteristic features of retinal detachment, including a hyperechoic membrane visible in the vitreous cavity. The attention maps focus on areas where the detached retina is most evident, particularly showing the characteristic \"V-shaped\" or \"seagull\" configuration...\n\n2. **Anatomical Structures:**\nThe heatmaps emphasize the superior region, where the detachment is most pronounced. Key anatomical landmarks include the optic nerve insertion point and the extent of retinal elevation...\n\n3. **Subtype Classification: Macula Detached:**\nThe spatial attention patterns indicate macular involvement, as evidenced by the extension of highlighted features into the posterior pole region...\n\n4. **Motion Analysis:**\nAcross the selected frames, we observe characteristic floating motion of the detached retina, particularly evident in frames 12 and 18 where the membrane shows dynamic movement...\n\n5. **Clinical Recommendation:**\nUrgent ophthalmology referral for surgical intervention is recommended given the macular involvement..."
}
```

## Key Features

### 1. Multi-Class Hierarchical Diagnosis

Based on ERDES dataset structure (`balanced_split_desc.csv`):

```python
outputs = {
    'diagnostic': ['non_rd', 'rd'],  # 2 classes
    'subtype': ['normal', 'macula_intact', 'macula_detached', 'pvd'],  # 4 classes
    'anatomical': ['N/A', 'TD', 'ND', 'Bilateral', 'SD', 'ID']  # 6 classes
}
```

**Class Definitions**:
- **diagnostic_class**: 
  - `non_rd`: Non-Retinal Detachment (includes normal and PVD)
  - `rd`: Retinal Detachment
  
- **subtype**:
  - `normal`: Normal healthy eye
  - `macula_intact`: RD with macula attached
  - `macula_detached`: RD with macula detached  
  - `pvd`: Posterior Vitreous Detachment
  
- **anatomical_subclass**:
  - `N/A`: Not applicable (for normal/pvd cases)
  - `TD`: Total Detachment
  - `ND`: Nasal Detachment
  - `Bilateral`: Bilateral detachment
  - `SD`: Superior Detachment
  - `ID`: Inferior Detachment

### 2. Explainability
- **Frame Importance**: Which temporal segments are diagnostic
- **Spatial Attention**: Which regions in each frame are important
- **Heatmap Overlays**: Visual explanation for clinicians

### 3. Contrastive Learning
- Ensures VLM uses heatmap information
- Prevents reliance solely on frame content
- Improves reliability of explanations

### 4. Clinical Reasoning
- Structured explanations
- References to anatomical structures
- Motion pattern analysis
- Differential diagnoses
- Clinical recommendations

## Customization

### Modify Class Labels

Edit `vlm_data_preparation.py`:

```python
self.diagnostic_labels = {
    0: "Your Class 1",
    1: "Your Class 2",
    2: "Your Class 3"
}
```

### Adjust Number of Important Frames

```python
preparator = VLMDataPreparator(model, top_k_frames=7)  # Default: 5
```

### Change Heatmap Colormap

```python
heatmap = preparator.generate_heatmap_overlay(
    frame, attention, colormap='hot'  # Options: 'jet', 'hot', 'viridis', etc.
)
```

### Modify Prompt Template

Edit `create_prompt_with_predictions()` in `vlm_data_preparation.py`

## Performance Considerations

### Memory Requirements
- **Classifier**: ~2GB GPU memory
- **Qwen 2.5 VL (4-bit)**: ~8GB GPU memory
- **Training (batch_size=2)**: ~16GB GPU memory

### Training Time
- **Data Preparation**: ~1-2 min per 100 videos
- **VLM Finetuning**: ~2-4 hours for 3 epochs (200 samples)

### Inference Time
- **Classifier**: ~50ms per video
- **VLM**: ~2-5 seconds per diagnosis

## Troubleshooting

### Out of Memory
```python
# Use smaller batch size
batch_size = 1

# Use 4-bit quantization
load_in_4bit = True

# Reduce number of frames
top_k_frames = 3
```

### Poor VLM Reasoning
```python
# Increase contrastive weight
contrastive_weight = 0.5

# More training epochs
num_epochs = 5

# Better prompts - edit template in vlm_data_preparation.py
```

### Heatmaps Not Used
- Increase `contrastive_weight`
- Check that heatmaps are visually distinct
- Verify contrastive samples have random attention

## Citation

If you use this system, please cite:

```bibtex
@article{ozkuterdes,
  title={ERDES: A Benchmark Video Dataset for Retinal Detachment and Macular Status Classification in Ocular Ultrasound},
  author={Ozkut, Yasemin and Navard, Pouyan and Adhikari, Srikar and Situ-LaCasse, Elaine and Acu{\~n}a, Josie and Yarnish, Adrienne A and Yilmaz, Alper},
  journal={arXiv preprint arXiv:2508.04735},
  year={2025}
}

@article{qwen2vl,
  title={Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution},
  author={Wang, Peng and others},
  journal={arXiv preprint},
  year={2024}
}
```

## License

This code is part of the ERDES project research.
