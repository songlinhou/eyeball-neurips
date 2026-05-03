# Fix: Ground Truth Text ONLY Uses CSV Summaries (No Templates Allowed)

## Problem

During VLM finetuning, the ground truth text was showing a generic template instead of the actual expert clinical summaries from `balanced_split_desc.csv`:

```
**Primary Diagnosis: rd**

1. **Visual Features Supporting Diagnosis:**
   The highlighted regions show key diagnostic features consistent with rd...
```

This was a **template response**, not the real expert summary from your CSV.

## Root Cause

The data pipeline wasn't passing the `summary` and `diagnosis_text` fields from the CSV through to the VLM training samples.

**Missing data flow:**
```
CSV → ERDESDataset → metadata → ground_truth → VLM sample
      ❌ summary not included in metadata
```

## Solution

Updated three files to properly pass the expert summaries:

### 1. `erdes_dataset.py` - Include summary in metadata

```python
metadata = {
    'clip_id': row['clip_id'],
    'file_path': row['file_path'],
    'diagnostic_class': row['diagnostic_class'],
    'subtype': row['subtype'],
    'anatomical_subclass': anatomical_str,
    'fps': row['fps'],
    'frame_count': row['frame_count'],
    'duration_seconds': row['duration_seconds'],
    'summary': row.get('summary', ''),  # ✓ Added
    'diagnosis_text': row.get('diagnosis_text', '')  # ✓ Added
}
```

### 2. `train_llm.py` - Pass summary to VLM preparation

```python
ground_truth = {
    'diagnostic': metadata['diagnostic_class'],
    'subtype': metadata['subtype'],
    'anatomical': metadata.get('anatomical_subclass', 'N/A'),
    'summary': metadata.get('summary', ''),  # ✓ Added
    'diagnosis_text': metadata.get('diagnosis_text', '')  # ✓ Added
}
```

### 3. `vlm_data_preparation.py` - Save summary at top level

```python
sample = {
    'video_id': video_id,
    'predictions': {...},
    'frame_indices': frame_indices.tolist(),
    'importance_scores': importance_scores.tolist(),
    'frame_paths': frame_paths,
    'heatmap_paths': heatmap_paths,
    'prompt': prompt,
    'ground_truth': ground_truth,
    'summary': ground_truth.get('summary', '') if ground_truth else '',  # ✓ Added
    'diagnosis_text': ground_truth.get('diagnosis_text', '') if ground_truth else ''  # ✓ Added
}
```

## What You'll See Now

After re-running data preparation, the ground truth will show the **actual expert summaries** from your CSV:

```
================================================================================
SAMPLE 0 - Video ID: 102769_00036
Is Contrastive: False
================================================================================
GROUND TRUTH TEXT:
--------------------------------------------------------------------------------
The ocular ultrasound video presents a clear case of retinal detachment 
located on the temporal side of the eye. The video consistently shows a 
membranous structure, identified as the detached retina, floating within 
the vitreous cavity. Importantly, the macula remains unaffected throughout 
the video, maintaining its attachment and thus preserving the patient's 
central vision. The detachment is characterized by distinct linear echoes, 
predominantly on the temporal side, suggesting a localized separation in 
that quadrant. Despite the retinal detachment, the preservation of the 
macula supports a favorable prognosis for central vision...

**Structured Diagnosis:**
<diagnostic>rd</diagnostic>
<subtype>macula_intact</subtype>
<anatomical>TD</anatomical>
================================================================================
```

## How to Apply the Fix

### Option 1: Re-run Full Training (Recommended)

This will regenerate all VLM data with correct ground truth:

```bash
# Delete old VLM data cache
rm -rf ./checkpoints/vlm_finetuned/vlm_data

# Re-run training from scratch
bash run_training.sh
```

### Option 2: Re-run Data Preparation Only

If you want to keep the classifier:

```bash
# Delete VLM data but keep classifier
rm -rf ./checkpoints/vlm_finetuned/vlm_data

# Re-run with --no-vlm to skip VLM training (just prepare data)
bash run_training.sh --no-vlm

# Then manually run VLM training
python train_llm.py \
    --classifier_checkpoint ./checkpoints/multiclass/best_model_weights.pth \
    --csv_path ../benchmarks/input/balanced_split_desc.csv \
    --data_root ../erdes \
    --output_dir ./checkpoints/vlm_finetuned \
    --skip_data_preparation  # Use the newly prepared data
```

### Option 3: Force Re-prepare Data

Use the `--force_prepare` flag if it exists, or delete the cache:

```bash
rm -rf ./checkpoints/vlm_finetuned/vlm_data
bash run_training.sh
```

## Verification

After re-running, check that ground truth is correct:

1. **Look at console output** during training start:
   ```
   INFO:vlm_finetuning:GROUND TRUTH TEXT:
   The ocular ultrasound video presents a clear case...
   ```

2. **Check the JSON samples**:
   ```bash
   cat ./checkpoints/vlm_finetuned/vlm_data/train/all_samples.json | jq '.[0].summary'
   ```

3. **Verify it matches CSV**:
   ```bash
   # Get the summary from CSV for a specific clip_id
   grep "102769_00036" ../benchmarks/input/balanced_split_desc.csv
   ```

## Why This Matters

### Before (Template Response)
- Generic, non-specific clinical reasoning
- Same structure for all samples
- No real medical knowledge
- VLM learns to generate templates, not real diagnoses

### After (Expert Summaries)
- Actual expert clinical descriptions from your dataset
- Specific anatomical details and observations
- Real medical knowledge transfer
- VLM learns from clinical expertise

## Example Comparison

### Template (Wrong ❌)
```
**Primary Diagnosis: rd**

1. **Visual Features Supporting Diagnosis:**
   The highlighted regions show key diagnostic features consistent with rd.
   The attention maps focus on areas where characteristic patterns are most evident...
```

### Expert Summary (Correct ✓)
```
The ocular ultrasound video presents a clear case of retinal detachment 
located on the temporal side of the eye. The video consistently shows a 
membranous structure, identified as the detached retina, floating within 
the vitreous cavity. Importantly, the macula remains unaffected throughout 
the video, maintaining its attachment and thus preserving the patient's 
central vision. The detachment is characterized by distinct linear echoes, 
predominantly on the temporal side, suggesting a localized separation in 
that quadrant...

**Structured Diagnosis:**
<diagnostic>rd</diagnostic>
<subtype>macula_intact</subtype>
<anatomical>TD</anatomical>
```

## Strict Policy: No Templates Allowed

The code now **enforces** that only CSV summaries are used:

```python
# In vlm_finetuning.py
if 'summary' in sample and sample['summary'] and sample['summary'].strip():
    response = sample['summary']  # ✓ Use CSV summary
else:
    # ❌ FAIL - no template fallback allowed
    raise ValueError("No ground truth summary found. Only CSV summaries allowed.")
```

If a sample is missing the `summary` field, training will **fail immediately** with a clear error message instead of silently using a template.

## Summary

✓ **Fixed**: `erdes_dataset.py` now includes `summary` and `diagnosis_text` in metadata  
✓ **Fixed**: `train_llm.py` now passes these fields to VLM preparation  
✓ **Fixed**: `vlm_data_preparation.py` now saves them in samples  
✓ **Enforced**: Template responses are completely disabled - will raise error if CSV summary missing  
✓ **Action Required**: Re-run training to regenerate VLM data with correct ground truth

The VLM will now learn **ONLY** from real expert clinical knowledge from your CSV - no templates allowed!
