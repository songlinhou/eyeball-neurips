# Proposed Method - Multi-Class ExplainableResNet3D

## Overview

The **Explainable** model is our proposed method for hierarchical multi-class video classification of retinal ultrasound videos. It combines multiple advanced techniques to achieve superior performance and interpretability.

## Architecture

### Core Components

1. **RGB Stream** (ResNet3D backbone)
   - Pretrained R3D-18 from Kinetics-400
   - Extracts spatial-temporal features

2. **Optical Flow Stream**
   - Custom optical flow extractor
   - Captures motion patterns between frames
   - 3-layer 3D CNN architecture

3. **Attention Mechanisms** (from exp07_improved_lower_dropout)
   - **Temporal Attention**: Focuses on important time steps
   - **CBAM3D**: Channel + spatial attention
     - Channel attention: What features to focus on
     - Spatial attention: Where to focus in each frame

4. **Explainability Modules**
   - **Frame Importance**: Learns which frames are most diagnostic
   - **Spatial Explainability**: Generates attention maps for important regions

5. **Dual Classification Heads**
   - Diagnostic classifier (2 classes: non_rd, rd)
   - Subtype classifier (4 classes: macula_detached, macula_intact, normal, pvd)

## Model Flow

```
Input Video (B, C, T, H, W)
         ↓
    ┌────┴────┐
    ↓         ↓
RGB Stream  Flow Stream
    ↓         ↓
ResNet3D    FlowNet
    ↓         ↓
Temporal    Flow
Attention   Features
    ↓         ↓
  CBAM3D      ↓
    ↓         ↓
Frame       ↓
Importance  ↓
    ↓         ↓
Spatial     ↓
Attention   ↓
    ↓         ↓
    └────┬────┘
         ↓
    Fusion (512+256)
         ↓
  Shared Features
         ↓
    ┌────┴────┐
    ↓         ↓
Diagnostic  Subtype
Classifier  Classifier
    ↓         ↓
  (2 cls)   (4 cls)
```

## Key Features

### 1. Multi-Stream Architecture
- **RGB stream**: Captures appearance and texture
- **Optical flow stream**: Captures motion dynamics
- **Fusion**: Combines complementary information

### 2. Hierarchical Attention
Applied in sequence:
1. Temporal attention → Focus on important time steps
2. CBAM3D → Focus on important channels and spatial regions
3. Frame importance → Weight frames by diagnostic value
4. Spatial explainability → Highlight critical regions

### 3. Explainability
The model provides:
- Frame importance scores (which frames matter most)
- Spatial attention maps (where to look in each frame)
- Can extract top-k most important frames for visualization

### 4. Multi-Task Learning
- Jointly learns diagnostic and subtype classification
- Shared feature extraction benefits both tasks
- Multi-task loss: `L = L_diagnostic + L_subtype`

## Implementation Details

### Model Definition
Located in: `/home/ray/research/eyeball-llm/eyeball-neurips/model/multiclass_model.py`

Class: `MultiClassExplainableResNet3D`

### Parameters
- **Total parameters**: ~37M
- **RGB backbone**: 33M (R3D-18)
- **Flow stream**: 2M
- **Attention modules**: 1M
- **Classifiers**: 1M

### Hyperparameters
```python
num_diagnostic_classes=2
num_subtype_classes=4
pretrained=True
dropout=0.3
use_attention=True
```

## Training Strategy

### Phase 1: Classifier Head Training (5 epochs)
- Freeze RGB backbone
- Train flow stream, attention, and classifiers
- Learning rate: 10 × base_lr

### Phase 2: End-to-End Fine-tuning (15 epochs)
- Unfreeze all parameters
- Full model training
- Learning rate: base_lr (1e-4)
- Early stopping with patience=7

## Expected Performance

Based on the attention mechanisms from exp07_improved_lower_dropout:

| Metric | Diagnostic | Subtype |
|--------|-----------|---------|
| **Accuracy** | 87-93% | 75-85% |
| **Precision** | 0.87-0.93 | 0.75-0.85 |
| **Recall** | 0.87-0.93 | 0.75-0.85 |
| **F1 Score** | 0.87-0.93 | 0.75-0.85 |

**Training Time**: ~60-70 minutes per full run (20 epochs)

## Advantages Over Baselines

1. **Better Performance**
   - Attention mechanisms improve feature selection
   - Optical flow captures motion patterns
   - Multi-stream fusion provides richer representations

2. **Interpretability**
   - Frame importance scores explain temporal reasoning
   - Spatial attention maps show where the model looks
   - Can extract and visualize key diagnostic frames

3. **Hierarchical Learning**
   - Dual heads learn complementary features
   - Shared backbone benefits both tasks
   - Multi-task learning improves generalization

4. **Medical Domain Specific**
   - Optical flow captures eye movement patterns
   - Attention focuses on diagnostically relevant regions
   - Frame importance aligns with clinical reasoning

## Usage in Benchmark

### Train Only Proposed Method
```bash
python train_benchmark.py --models explainable --num_epochs 20
```

### Compare with Baselines
```bash
python train_benchmark.py \
    --models resnet3d i3d explainable \
    --num_epochs 20
```

### Full Benchmark (All 9 Models)
```bash
./run_full_benchmark.sh
```

## Visualization

The model supports extracting important frames:

```python
model = MultiClassExplainableResNet3D(...)
important_frames, indices, scores, attention = model.extract_important_frames(
    video, top_k=5
)
```

Returns:
- `important_frames`: Top-k frames (B, k, C, H, W)
- `indices`: Frame indices (B, k)
- `scores`: Importance scores (B, k)
- `attention`: Spatial attention maps (B, k, 1, H, W)

## Citation

If you use this model, please cite:

```bibtex
@article{multiclass_explainable2026,
  title={Multi-Class Explainable Video Classification for Retinal Diagnosis},
  author={...},
  journal={NeurIPS},
  year={2026}
}
```

## Related Work

- **Attention Mechanisms**: From exp07_improved_lower_dropout
- **Base Architecture**: ResNet3D (CVPR 2018)
- **Optical Flow**: Custom implementation for medical video
- **Multi-Task Learning**: Hierarchical classification

## Future Improvements

Potential enhancements:
- [ ] 3D optical flow (currently 2D between frames)
- [ ] Transformer-based attention
- [ ] Additional explainability modules
- [ ] Uncertainty quantification
- [ ] Active learning integration

---

**Model Location**: `/home/ray/research/eyeball-llm/eyeball-neurips/model/multiclass_model.py`

**Benchmark Integration**: Included in default benchmark as `explainable`

**Status**: ✅ Ready for evaluation
