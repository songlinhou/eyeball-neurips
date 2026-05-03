# METHODOLOGY.md Update Summary

## Overview

Updated the methodology document to reflect the complete two-stage framework:
1. **Multi-Class Explainable Video Classifier** (Stage 1)
2. **Vision-Language Model Integration** (Stage 2)

## Major Changes

### 1. Title and Abstract Updated

**Before**: "Explainable Optical Flow ResNet3D for Medical Video Classification"
- Single-task binary classification
- Focus on CNN architecture only

**After**: "Explainable Multi-Class Video Classification with Vision-Language Model Integration"
- Multi-task hierarchical classification (diagnostic + subtype)
- Two-stage framework with VLM integration
- Emphasis on clinical reasoning generation

### 2. Problem Formulation (Section 3.1)

**Added**:
- **Section 3.1.1**: Multi-Class Hierarchical Classification
  - Formal definition of multi-task learning objective
  - Diagnostic classification: 2 classes (non_rd, rd)
  - Subtype classification: 4 classes (normal, macula_intact, macula_detached, pvd)
  - Explanation of why anatomical classification is excluded
  - Clinical alignment justification

**Mathematical Formulation**:
```
f: R^(T×H×W×3) → R^(C_diag) × R^(C_sub)
```
where C_diag = 2, C_sub = 4

### 3. Multi-Task Classification Architecture (Section 3.2.3)

**Updated**:
- Changed from single classifier to dual classification heads
- Shared feature representation with task-specific heads
- Added mathematical formulation for multi-task outputs

**New Equations**:
```
h_shared = Dropout(BN(ReLU(W_shared * z_fused + b_shared)))
y_diag = W_diag * h_shared + b_diag ∈ R^(B×2)
y_sub = W_sub * h_shared + b_sub ∈ R^(B×4)
```

**Benefits Highlighted**:
1. Shared representations
2. Task regularization
3. Parameter efficiency
4. Clinical workflow alignment

### 4. Multi-Task Loss Function (Section 3.3.2)

**Replaced**: Focal Loss for single-task
**With**: Multi-task Cross-Entropy Loss

**New Formulation**:
```
L_total = L_diag + λ_sub * L_sub
```
where λ_sub = 1.0 (equal weighting)

**Added**: Balanced Sampling Strategy
- Stratified train/test split by combined diagnostic-subtype labels
- Ensures balanced class representation
- Superior to class-weighted loss for small datasets

### 5. NEW SECTION: Vision-Language Model Integration (Section 3.4)

Completely new section covering the VLM pipeline:

#### 3.4.1 VLM Data Preparation

**Step 1: Important Frame Extraction**
- Top-K frames based on frame attention scores
- K = 5 for efficiency
- Mathematical formulation with TopK operation

**Step 2: Attention Heatmap Generation**
- Upsample spatial attention to original resolution
- Overlay on frames with jet colormap
- Create attention-highlighted visualizations

**Step 3: Prediction-Conditioned Prompt Generation**
- Structured prompts with classifier predictions
- Include diagnostic class, subtype, and confidence scores
- Format: "The AI model predicts: ..."

**Step 4: Contrastive Sample Creation**
- Positive samples: Attention-highlighted frames
- Negative samples: Original frames without highlights
- Ensures VLM utilizes attention information

#### 3.4.2 VLM Finetuning

**Model**: Qwen 2.5 VL-7B
- Vision encoder + 7B language model
- Cross-modal fusion with attention

**LoRA Configuration**:
```
W' = W_0 + ΔW = W_0 + B*A
```
- Rank r = 16
- Alpha α = 32
- Dropout = 0.05

**4-bit Quantization**:
- Reduces memory: 28GB → 7GB
- Enables training on consumer GPUs

**Training Objective**:
```
L_VLM = -1/N Σ Σ log P(w_t | w_<t, V_1:K, prompt)
```

**Hyperparameters**:
- Learning rate: 2×10^-5
- Batch size: 2
- Epochs: 10
- Warmup: 100 steps

#### 3.4.3 Inference Pipeline

6-step pipeline:
1. Video classification → predictions + attention
2. Frame selection (top-K)
3. Heatmap overlay
4. Prompt construction
5. Clinical reasoning generation
6. Structured output

**Advantages**:
- Modularity
- Efficiency
- Explainability
- Clinical alignment

### 6. Dataset Description Updated (Section 3.5)

**Before**: "macula detached vs. intact split, ~200 videos"

**After**: Complete ERDES statistics
- 5,383 total videos
- Hierarchical labels (diagnostic, subtype, anatomical)
- Stratified 80/20 train/test split
- Detailed class distribution

### 7. Evaluation Metrics Enhanced (Section 3.5)

**Added**:
- Per-task metrics (diagnostic and subtype)
- VLM evaluation metrics:
  - Factual consistency
  - Clinical relevance
  - Attention utilization
- Clarified attention map dual purpose (validation + VLM input)

### 8. New References Added

**[11]** Bai et al. (2023) - Qwen-VL model
**[12]** Hu et al. (2021) - LoRA
**[13]** Dettmers et al. (2023) - QLoRA (4-bit quantization)

### 9. Appendix Tables Updated

**Table 3**: Updated to Multi-Task Classification Architecture
- Added separate diagnostic and subtype heads
- Shows shared FC layer before task-specific outputs

**Table 5**: NEW - VLM Hyperparameters
- Complete VLM training configuration
- LoRA parameters
- Quantization settings
- Contrastive learning flag

**Table 6**: NEW - Dataset Statistics
- Complete ERDES breakdown
- Class distributions
- Train/test split details

## Key Improvements

### 1. Clinical Relevance
- Hierarchical classification matches diagnostic workflow
- VLM generates human-readable explanations
- Attention maps provide visual evidence

### 2. Technical Rigor
- Formal mathematical notation for all components
- Clear justification for design choices
- Complete hyperparameter specifications

### 3. Reproducibility
- Detailed training procedures for both stages
- Exact hyperparameters in appendix tables
- Dataset statistics and split methodology

### 4. Explainability
- Multi-level interpretability:
  - Frame importance (temporal)
  - Spatial attention (spatial)
  - Natural language reasoning (VLM)

### 5. Efficiency
- Multi-task learning reduces parameters
- LoRA + 4-bit quantization enables VLM training
- Two-stage approach allows modular updates

## Methodology Flow Diagram

```
Input Video (T×H×W×3)
    ↓
┌─────────────────────────────────────┐
│  Stage 1: Multi-Class Classifier    │
├─────────────────────────────────────┤
│ • RGB Stream (R3D-18)               │
│ • Optical Flow Stream               │
│ • Frame + Spatial Attention         │
│ • Feature Fusion                    │
│ • Dual Classification Heads         │
└─────────────────────────────────────┘
    ↓
Outputs:
• Diagnostic: {non_rd, rd}
• Subtype: {normal, macula_intact, macula_detached, pvd}
• Frame Importance Scores
• Spatial Attention Maps
    ↓
┌─────────────────────────────────────┐
│  Stage 2: VLM Integration           │
├─────────────────────────────────────┤
│ • Extract Top-5 Important Frames    │
│ • Generate Attention Heatmaps       │
│ • Create Prediction Prompts         │
│ • Qwen 2.5 VL (LoRA + 4-bit)       │
└─────────────────────────────────────┘
    ↓
Final Output:
• Diagnostic + Subtype Predictions
• Confidence Scores
• Attention-Highlighted Frames
• Clinical Reasoning Text
```

## Summary Statistics

### Classifier (Stage 1)
- **Parameters**: ~40M
- **FLOPs**: ~45G
- **Training Time**: 4-6 hours (50 epochs)
- **Input**: 32 frames @ 224×224
- **Outputs**: 2 tasks (diagnostic + subtype)

### VLM (Stage 2)
- **Base Model**: Qwen 2.5 VL-7B
- **Trainable Params**: ~16M (LoRA only)
- **Memory**: ~7GB (with 4-bit quantization)
- **Training Time**: ~40-50 hours (10 epochs)
- **Input**: 5 frames + prompt
- **Output**: Clinical reasoning text

### Dataset
- **Total Videos**: 5,383
- **Train**: 4,306 (80%)
- **Test**: 1,077 (20%)
- **Classes**: 2 diagnostic × 4 subtypes = 8 combinations
- **Split Strategy**: Stratified by combined labels

## Files Modified

1. **METHODOLOGY.md** - Complete rewrite with VLM integration
2. **Tables in Appendix** - Updated for multi-task + added VLM tables
3. **References** - Added 3 new citations for VLM/LoRA

## Next Steps

The methodology now fully documents:
✅ Multi-class hierarchical classification
✅ Balanced sampling strategy
✅ VLM data preparation pipeline
✅ LoRA finetuning with 4-bit quantization
✅ Complete inference pipeline
✅ Evaluation metrics for both stages
✅ Reproducible hyperparameters

Ready for:
- Experimental results section
- Ablation studies
- Clinical validation
- Publication submission
