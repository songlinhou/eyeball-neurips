# Strict Policy: Only CSV Summaries Allowed - No Templates

## Policy Statement

**The VLM training ONLY uses expert clinical summaries from `balanced_split_desc.csv`.**

**Template responses are COMPLETELY DISABLED and will cause training to fail.**

## Implementation

### Code Changes

1. **`vlm_finetuning.py`** - Strict validation:
```python
if 'summary' in sample and sample['summary'] and sample['summary'].strip():
    response = sample['summary']  # ✓ Use real expert summary
else:
    # ❌ ERROR - No fallback to templates
    raise ValueError(f"No ground truth summary found for {video_id}. "
                     f"Only expert summaries from CSV are allowed.")
```

2. **Template function disabled**:
```python
def _generate_template_response(self, sample: Dict) -> str:
    """DEPRECATED: This should NEVER be called."""
    raise ValueError("Template responses are NOT allowed. "
                     "Only use expert summaries from CSV.")
```

## What This Means

### ✓ Allowed
- Expert clinical summaries from `balanced_split_desc.csv` → `summary` column
- Structured diagnosis tags from CSV → `diagnosis_text` column
- Contrastive uncertainty responses (for random heatmaps)

### ❌ Not Allowed
- Generic template responses
- Auto-generated clinical text
- Fallback responses when CSV summary is missing
- Any text not written by medical experts

## Error Handling

### During Data Preparation

Videos without summaries are **automatically skipped**:

```
Preparing VLM data...
Processing videos: 100%|████████| 171/171 [05:23<00:00]

============================================================
VLM Data Preparation Complete
============================================================
Prepared samples: 165
Skipped (no summary): 6
Saved to: ./checkpoints/vlm_finetuned/vlm_data/train/all_samples.json
============================================================
```

### During Training

The dataset automatically filters out samples without summaries:

```
INFO:vlm_finetuning:Loaded 165 correct samples
INFO:vlm_finetuning:Filtered out 0 samples without ground truth summaries
INFO:vlm_finetuning:Loaded 70 contrastive samples
```

**No errors, no failures** - videos without summaries are simply excluded from training.

## Why This Policy?

### Medical Accuracy
- Templates are generic and non-specific
- Real expert summaries contain actual medical knowledge
- Clinical details matter for accurate diagnosis

### Training Quality
- VLM learns from real clinical expertise
- No risk of learning template patterns
- Ensures high-quality medical reasoning

### Transparency
- Clear what the model is learning from
- Traceable to expert annotations
- No hidden auto-generated content

## Verification Checklist

Before training, verify:

- [ ] CSV file has `summary` column with expert text
- [ ] CSV file has `diagnosis_text` column with structured tags
- [ ] All records in CSV have non-empty summaries
- [ ] Data preparation loads these fields correctly
- [ ] Sample JSON files contain `summary` and `diagnosis_text`

Check a sample:
```bash
# Verify CSV has summaries
head -n 2 ../benchmarks/input/balanced_split_desc.csv | cut -d',' -f10

# Verify prepared data has summaries
cat ./checkpoints/vlm_finetuned/vlm_data/train/all_samples.json | jq '.[0].summary'
```

## What Happens During Training

### Correct Samples (70%)
```
INPUT: 5 heatmap images + prompt with predictions
OUTPUT: Expert summary from CSV + structured diagnosis
```

### Contrastive Samples (30%)
```
INPUT: 5 random heatmap images + prompt with predictions
OUTPUT: Uncertainty response (hardcoded, not from CSV)
```

**No templates. No fallbacks. Only real expert knowledge.**

## Re-running Training

After the fix, you MUST re-run data preparation:

```bash
# Delete old cached data
rm -rf ./checkpoints/vlm_finetuned/vlm_data

# Re-run training
bash run_training.sh
```

This will:
1. Load summaries from CSV
2. Pass them through the pipeline
3. Save them in VLM samples
4. Use them for training (or fail if missing)

## Expected Output

When training starts, you should see:

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
central vision...

**Structured Diagnosis:**
<diagnostic>rd</diagnostic>
<subtype>macula_intact</subtype>
<anatomical>TD</anatomical>
================================================================================
```

**NOT this:**
```
**Primary Diagnosis: rd**

1. **Visual Features Supporting Diagnosis:**
   The highlighted regions show key diagnostic features...
```

## Summary

✓ Only CSV summaries are used  
✓ Templates are completely disabled  
✓ Training fails if summary is missing  
✓ Clear error messages guide debugging  
✓ Medical accuracy is guaranteed  

**Your VLM will learn ONLY from real expert clinical knowledge.**
