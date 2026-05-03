# VLM Finetuning: Input and Output Format

This document explains exactly what goes into the VLM during finetuning and what comes out.

## Overview

The VLM (Qwen 2.5 VL) is finetuned to generate **clinical reasoning** that explains why the classifier made certain predictions based on visual attention patterns.

---

## Input Format

### Visual Input: 5 Heatmap Images

For each video, the top-5 most important frames are extracted with attention heatmaps overlaid:

```
Frame 1 (heatmap)  Frame 2 (heatmap)  Frame 3 (heatmap)  Frame 4 (heatmap)  Frame 5 (heatmap)
     [IMAGE]            [IMAGE]            [IMAGE]            [IMAGE]            [IMAGE]
```

**What the heatmaps show:**
- Red/yellow regions = High attention (important for diagnosis)
- Blue/green regions = Low attention
- These highlight which anatomical structures the classifier focused on

### Text Input: Prompt with Predictions

```
The AI model has analyzed this ocular ultrasound video and made the following predictions:

Primary Diagnosis: RD (confidence: 95.23%)
Subtype Classification: Macula Detached (confidence: 87.45%)

Based on the highlighted regions in the images (shown as heatmaps), please explain:
1. Why this diagnosis is likely based on the visual features you observe
2. What specific anatomical structures or patterns support this classification
3. How the motion patterns (if visible) contribute to the diagnosis
4. Any potential differential diagnoses to consider

Please provide a detailed clinical reasoning for these predictions.
```

---

## Output Format (Ground Truth During Training)

### For Correct Samples (70% of training data)

The VLM learns to generate **expert clinical summaries** from your CSV file:

**Example from `balanced_split_desc.csv`:**
```
Based on the analysis of the highlighted regions in these ultrasound frames:

The retinal detachment is clearly visible with the characteristic V-shaped 
membrane floating in the vitreous cavity. The macula appears detached, 
indicated by the separation of the retinal layers in the posterior pole. 
The highlighted regions correctly focus on:

1. The detached retinal membrane showing typical undulating motion
2. The subretinal space with characteristic echo-free appearance
3. The point of attachment near the optic nerve head
4. The macular region showing clear separation from the RPE

The temporal progression across frames demonstrates the dynamic nature of 
the detachment, with the membrane showing movement consistent with eye 
motion. This supports the diagnosis of rhegmatogenous retinal detachment 
with macular involvement.

**Structured Diagnosis:**
<diagnostic>RD</diagnostic>
<subtype>Macula Detached</subtype>
<anatomical>superior</anatomical>
```

### For Contrastive Samples (30% of training data)

When shown **random/incorrect heatmaps**, the VLM learns to express uncertainty:

```
I notice the highlighted regions in these images appear random or 
inconsistent with typical diagnostic patterns. The heatmap does not 
clearly indicate specific anatomical structures that would support 
the predicted diagnosis.

To provide accurate clinical reasoning, I would need:
1. More focused attention on relevant anatomical landmarks
2. Clearer visualization of the pathological features
3. Consistent highlighting across frames showing the progression

Without reliable visual guidance, I cannot confidently explain why 
this specific diagnosis was made based solely on these highlighted regions.
```

**Purpose:** This teaches the VLM to actually **use** the heatmap information rather than just memorizing text patterns.

---

## Complete Training Sample Structure

Each training sample in the JSON file looks like:

```json
{
  "video_id": "164267_02030_correct",
  
  "frame_paths": [
    "./vlm_data/164267_02030_frame_0_idx5.jpg",
    "./vlm_data/164267_02030_frame_1_idx12.jpg",
    "./vlm_data/164267_02030_frame_2_idx18.jpg",
    "./vlm_data/164267_02030_frame_3_idx24.jpg",
    "./vlm_data/164267_02030_frame_4_idx29.jpg"
  ],
  
  "heatmap_paths": [
    "./vlm_data/164267_02030_heatmap_0_idx5.jpg",
    "./vlm_data/164267_02030_heatmap_1_idx12.jpg",
    "./vlm_data/164267_02030_heatmap_2_idx18.jpg",
    "./vlm_data/164267_02030_heatmap_3_idx24.jpg",
    "./vlm_data/164267_02030_heatmap_4_idx29.jpg"
  ],
  
  "predictions": {
    "diagnostic": "RD",
    "diagnostic_confidence": 0.9523,
    "subtype": "Macula Detached",
    "subtype_confidence": 0.8745
  },
  
  "frame_indices": [5, 12, 18, 24, 29],
  "importance_scores": [0.22, 0.19, 0.15, 0.13, 0.11],
  
  "prompt": "The AI model has analyzed this ocular ultrasound video...",
  
  "summary": "Expert clinical description from CSV (ground truth)",
  
  "diagnosis_text": "<diagnostic>RD</diagnostic><subtype>Macula Detached</subtype>",
  
  "is_contrastive": false
}
```

---

## Training Process

### Step 1: Data Preparation
```python
# For each video in your dataset:
1. Run classifier → Get predictions (RD/Non-RD, subtype)
2. Extract top-5 important frames using attention
3. Generate heatmap overlays showing attention
4. Create prompt with predictions
5. Load ground truth summary from CSV
6. Save as JSON sample
```

### Step 2: VLM Finetuning
```python
# For each training sample:
INPUT:
  - Images: 5 heatmap overlays
  - Text: Prompt with classifier predictions
  
TARGET OUTPUT:
  - If correct heatmaps: Expert clinical summary from CSV
  - If contrastive (random heatmaps): Uncertainty response

LOSS:
  - Language modeling loss (predict next token)
  - Contrastive loss (different outputs for correct vs random heatmaps)
```

### Step 3: Inference
```python
# For a new video:
INPUT:
  - 5 heatmap images from classifier
  - Prompt with predictions
  
OUTPUT (Generated by VLM):
  "Based on the highlighted regions, the diagnosis of RD with 
   macular detachment is supported by the following features:
   
   1. The V-shaped membrane visible in frames 2-4...
   2. The subretinal space showing echo-free appearance...
   3. The macular region demonstrating clear separation...
   
   The temporal progression indicates..."
```

---

## Key Insights

### Why This Design?

1. **Grounded in Visual Evidence**: The VLM must explain based on what the classifier "saw" (attention maps)

2. **Faithfulness via Contrastive Learning**: By training on both correct and random heatmaps, the VLM learns to:
   - Use the visual information (not just memorize text)
   - Express uncertainty when heatmaps don't make sense

3. **Expert Knowledge Transfer**: Ground truth summaries from your CSV provide clinical expertise

4. **Structured Output**: The `<diagnostic>` tags enable parsing for downstream systems

### What Makes This Different from Standard VLM?

| Standard VLM | Your VLM |
|--------------|----------|
| Just describe what you see | Explain why the AI made this diagnosis |
| No attention guidance | Heatmaps show what AI focused on |
| Generic image understanding | Medical domain-specific reasoning |
| No faithfulness guarantee | Contrastive learning ensures attention usage |

---

## Example End-to-End Flow

### Training Example

```
VIDEO: 164267_02030.mp4 (Retinal Detachment with Macula Detached)

CLASSIFIER OUTPUT:
  ✓ Diagnostic: RD (95.2%)
  ✓ Subtype: Macula Detached (87.5%)
  ✓ Top frames: [5, 12, 18, 24, 29]
  ✓ Attention maps: Focused on retinal membrane and macula

VLM INPUT:
  📸 5 heatmap images
  📝 "The AI model predicts: RD (95.2%), Macula Detached (87.5%)..."

VLM TARGET OUTPUT (from CSV):
  "The retinal detachment is clearly visible with characteristic 
   V-shaped membrane. The macula appears detached, indicated by 
   separation of retinal layers in the posterior pole..."

TRAINING:
  VLM learns to generate this clinical reasoning when shown 
  these specific heatmap patterns
```

### Inference Example

```
NEW VIDEO: test_video.mp4

CLASSIFIER OUTPUT:
  ✓ Diagnostic: Non-RD (92.1%)
  ✓ Subtype: PVD (88.3%)
  ✓ Top frames: [3, 8, 15, 21, 28]
  ✓ Attention maps: Focused on vitreous cavity

VLM INPUT:
  📸 5 heatmap images
  📝 "The AI model predicts: Non-RD (92.1%), PVD (88.3%)..."

VLM GENERATED OUTPUT:
  "Based on the highlighted regions, the diagnosis of posterior 
   vitreous detachment (PVD) is supported by:
   
   1. The vitreous cavity shows characteristic separation from 
      the posterior retinal surface
   2. The Weiss ring is visible in frames 2-3, indicating 
      complete PVD
   3. No retinal detachment is evident - the retina remains 
      attached throughout
   4. The motion patterns show typical vitreous mobility
   
   This is consistent with uncomplicated PVD without retinal 
   involvement."
```

---

## Data Sources

### Where Each Component Comes From

| Component | Source | Example |
|-----------|--------|---------|
| **Video** | `../erdes/clips/` | `Retinal_Detachment/164267_02030.mp4` |
| **Predictions** | Trained classifier | `RD (95.2%)` |
| **Attention Maps** | Classifier's attention modules | Heatmap highlighting retinal membrane |
| **Ground Truth Summary** | `balanced_split_desc.csv` → `summary` column | Expert clinical description |
| **Structured Diagnosis** | `balanced_split_desc.csv` → parsed from columns | `<diagnostic>RD</diagnostic>` |

### CSV Structure Used

```csv
clip_id,diagnostic_class,subtype,anatomical_subclass,summary
164267_02030,RD,Macula Detached,superior,"The retinal detachment is clearly visible..."
```

The `summary` column contains the expert-written clinical descriptions that become the VLM's training targets.

---

## Summary

**INPUT**: 5 heatmap images + prompt with classifier predictions  
**OUTPUT**: Clinical reasoning explaining why the diagnosis makes sense based on visual features  
**TRAINING**: Learn from expert summaries in CSV, with contrastive learning for faithfulness  
**GOAL**: Explainable AI that can justify its predictions using visual evidence
