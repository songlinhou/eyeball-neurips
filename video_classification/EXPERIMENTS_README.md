# Ocular Ultrasound Classifier - Enhanced Experiments

## Overview

This document describes the enhanced experiments for classifying ocular ultrasound videos into **Macula Intact** vs **Macula Detached** categories.

## New Features

### 1. **Explainability Modules**
All new models include built-in explainability features:

- **Frame Importance Scores**: Identifies which frames in the video are most critical for classification
- **Spatial Attention Maps**: Highlights important regions within each frame
- These features enable clinical interpretation and validation of model decisions

### 2. **Optical Flow Integration**
Motion-based features to capture temporal dynamics:

- **Optical Flow Extraction**: Learns motion patterns between consecutive frames
- **Dual-Stream Architecture**: Combines RGB appearance and motion information
- Particularly useful for detecting retinal movement patterns

### 3. **Temporal Shift Module (TSM)**
Memory-efficient temporal modeling:

- Shifts feature channels across time dimension
- Reduces memory footprint compared to full 3D convolutions
- Enables longer sequences or larger batch sizes

### 4. **Memory Optimization**
Fixes for CUDA out-of-memory errors:

- Reduced batch sizes for A100 GPU
- Memory cleanup between experiments and training phases
- Gradient accumulation support for effective larger batches

## Experiment Configurations

### Original Experiments (exp01-exp07)
- **exp01**: Improved baseline with attention
- **exp02**: Multi-scale feature extraction
- **exp03**: Auxiliary classifier
- **exp04**: 16 frames (faster training)
- **exp05**: 64 frames (more temporal context)
- **exp06**: Higher learning rate
- **exp07**: Lower dropout (0.3) - **BEST PERFORMER** ✓

### New Experiments (exp08-exp14)

Based on the successful exp07 configuration (dropout=0.3, lr=1e-4):

#### **exp08: Explainable Lower Dropout**
- Model: `ExplainableResNet3D`
- Features: Frame importance + spatial attention
- Frames: 32, Batch: 4
- **Use Case**: When interpretability is critical

#### **exp09: Optical Flow Lower Dropout**
- Model: `OpticalFlowResNet3D`
- Features: RGB + motion flow dual stream
- Frames: 32, Batch: 4
- **Use Case**: Capturing retinal motion patterns

#### **exp10: Explainable Flow Lower Dropout**
- Model: `ExplainableOpticalFlowResNet3D`
- Features: Combined optical flow + explainability
- Frames: 32, Batch: 3 (smaller due to dual stream)
- **Use Case**: Best of both worlds - motion + interpretability

#### **exp11: TSM Lower Dropout**
- Model: `TSMResNet3D`
- Features: Temporal shift + explainability
- Frames: 32, Batch: 4
- **Use Case**: Memory-efficient temporal modeling

#### **exp12: Explainable 16 Frames**
- Model: `ExplainableResNet3D`
- Features: Frame importance + spatial attention
- Frames: 16, Batch: 8 (faster training)
- **Use Case**: Quick iterations with interpretability

#### **exp13: Explainable Higher LR**
- Model: `ExplainableResNet3D`
- Features: Frame importance + spatial attention
- Frames: 32, Batch: 4, LR: 2e-4
- **Use Case**: Testing faster convergence

#### **exp14: Optical Flow 16 Frames**
- Model: `OpticalFlowResNet3D`
- Features: RGB + motion flow
- Frames: 16, Batch: 8
- **Use Case**: Fast motion-aware training

## Model Architectures

### ExplainableResNet3D
```
Input Video (B, 3, T, H, W)
    ↓
ResNet3D Backbone
    ↓
Frame Importance Module → Frame scores (B, T)
    ↓
Spatial Explainability → Attention maps (B, 1, T, H, W)
    ↓
Global Pooling + Classifier → Predictions (B, 2)
```

### OpticalFlowResNet3D
```
Input Video (B, 3, T, H, W)
    ↓
    ├─→ RGB Stream (ResNet3D) → RGB features
    └─→ Flow Extractor → Flow Stream → Motion features
            ↓
    Fusion (RGB + Motion)
            ↓
    Classifier → Predictions (B, 2)
```

### ExplainableOpticalFlowResNet3D
```
Input Video (B, 3, T, H, W)
    ↓
    ├─→ RGB Stream → Frame Importance → Spatial Attention → RGB features
    └─→ Flow Extractor → Flow Stream → Motion features
            ↓
    Fusion (RGB + Motion)
            ↓
    Classifier → Predictions (B, 2)
```

## Running Experiments

### Run All Experiments
```bash
cd /content/improved_classifier
python run_experiments.py
```

Results will be saved to: `/content/drive/MyDrive/EyeballProject/classifier_experiment/`

### Output Structure
```
classifier_experiment/
├── models/                      # Trained model weights
│   ├── exp08_explainable_lower_dropout_best.pth
│   ├── exp09_optical_flow_lower_dropout_best.pth
│   └── ...
├── checkpoints/                 # Training checkpoints
├── logs/                        # Training logs
├── plots/                       # Training curves and confusion matrices
├── results/                     # Summary CSV and JSON
│   ├── experiment_summary.csv
│   └── experiment_results_detailed.json
└── EXPERIMENT_REPORT.md        # Comprehensive report
```

## Extracting Explainability Features

For models with explainability (exp08, exp10, exp11, exp12, exp13):

```bash
python visualize_explainability.py \
    --model_path /content/drive/MyDrive/EyeballProject/classifier_experiment/models/exp08_explainable_lower_dropout_best.pth \
    --model_class explainable \
    --save_dir /content/drive/MyDrive/EyeballProject/classifier_experiment/explainability/exp08 \
    --num_samples 20
```

### Explainability Output
```
explainability/exp08/
├── sample_000/
│   ├── info.txt                 # Prediction details
│   ├── frame_importance.png     # Which frames matter most
│   └── spatial_attention.png    # Where the model looks
├── sample_001/
│   └── ...
└── ...
```

## Key Improvements Over Previous Run

1. **Memory Management**: 
   - Reduced batch sizes to prevent CUDA OOM
   - Added memory cleanup between experiments
   - Should run successfully on A100 80GB

2. **Explainability**:
   - Frame-level importance scores
   - Pixel-level attention maps
   - Clinical interpretability

3. **Motion Features**:
   - Optical flow for temporal dynamics
   - Better capture of retinal movement

4. **Efficiency**:
   - TSM for memory-efficient temporal modeling
   - 16-frame variants for faster iteration

## Expected Performance

Based on exp07 results (87.13% test accuracy):
- **exp08-exp11**: Expected 85-90% test accuracy with added interpretability
- **exp12, exp14**: Expected 80-85% (fewer frames, faster training)
- **exp13**: Expected 85-88% (higher LR may converge faster but less stable)

## Clinical Use Cases

1. **High Accuracy Needed**: Use exp07 or exp08
2. **Interpretability Required**: Use exp08, exp10, or exp13
3. **Motion Analysis**: Use exp09 or exp14
4. **Fast Inference**: Use exp12 (16 frames)
5. **Best Overall**: Use exp10 (motion + explainability)

## Monitoring Training

Watch for:
- **Phase 1** (epochs 1-5): Classifier head training, expect 50-70% val accuracy
- **Phase 2** (epochs 6-25): Full fine-tuning, expect 80-95% val accuracy
- **Early stopping**: Triggers after 7 epochs without improvement
- **Memory usage**: Should stay under 70GB on A100

## Troubleshooting

### CUDA Out of Memory
- Reduce batch_size in experiment config
- Use 16-frame variants (exp12, exp14)
- Enable gradient checkpointing (if needed)

### Poor Performance
- Check data augmentation is enabled
- Verify class weights are applied
- Ensure Phase 2 fine-tuning completes
- Try higher learning rate (exp13)

### No Explainability Output
- Only works with explainable models (exp08, exp10, exp11, exp12, exp13)
- Check model loaded correctly
- Verify `return_attention=True` parameter

## Citation

If you use these models in your research, please cite:
- ResNet3D: Hara et al. "Learning Spatio-Temporal Features with 3D Residual Networks"
- Temporal Shift Module: Lin et al. "TSM: Temporal Shift Module for Efficient Video Understanding"
- CBAM Attention: Woo et al. "CBAM: Convolutional Block Attention Module"
