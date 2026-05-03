# Automatic Skipping of Videos Without Ground Truth Summaries

## Policy

Videos that don't have expert clinical summaries in the CSV are **automatically skipped** during both data preparation and training.

**No errors. No failures. Just clean filtering.**

## Implementation

### 1. Data Preparation (`train_llm.py`)

Videos without summaries are skipped during VLM data preparation:

```python
# Check if summary exists - skip videos without ground truth
summary = metadata.get('summary', '').strip()
if not summary:
    skipped_count += 1
    continue  # Skip this video
```

### 2. Dataset Loading (`vlm_finetuning.py`)

Samples without summaries are filtered when loading the dataset:

```python
# Filter out samples without ground truth summaries
self.correct_samples = [
    s for s in all_samples 
    if not s.get('is_contrastive', False) 
    and s.get('summary') and s.get('summary').strip()
]
```

### 3. Runtime Safety (`vlm_finetuning.py`)

If a sample somehow gets through without a summary, it returns None:

```python
if 'summary' in sample and sample['summary'] and sample['summary'].strip():
    response = sample['summary']
else:
    logger.warning(f"Skipping video {video_id} - no ground truth summary")
    return None  # Skip this sample
```

## What You'll See

### During Data Preparation

```bash
Preparing VLM data...
Processing videos: 100%|████████████████| 171/171 [05:23<00:00]

============================================================
VLM Data Preparation Complete
============================================================
Prepared samples: 165
Skipped (no summary): 6
Saved to: ./checkpoints/vlm_finetuned/vlm_data/train/all_samples.json
============================================================
```

**6 videos were skipped** because they don't have summaries in the CSV.

### During Training

```bash
INFO:vlm_finetuning:Loaded 165 correct samples
INFO:vlm_finetuning:Filtered out 0 samples without ground truth summaries
INFO:vlm_finetuning:Loaded 70 contrastive samples
```

All samples loaded have valid summaries. No filtering needed at this stage because data prep already handled it.

## Benefits

### ✓ Clean Training Data
- Only videos with expert summaries are used
- No missing or incomplete data
- Consistent quality across all samples

### ✓ No Manual Intervention
- Automatic filtering at multiple stages
- No need to manually clean the CSV
- Works with any CSV structure

### ✓ Clear Reporting
- See exactly how many videos were skipped
- Understand your dataset composition
- Track data quality

### ✓ Robust Pipeline
- Multiple safety checks
- Graceful handling of missing data
- No training failures

## Checking Your Data

### Before Training

Check how many videos have summaries:

```bash
# Count non-empty summaries in CSV
awk -F',' 'NR>1 && $10!="" {count++} END {print "Videos with summaries:", count}' \
    ../benchmarks/input/balanced_split_desc.csv

# Count total videos
wc -l ../benchmarks/input/balanced_split_desc.csv
```

### After Data Preparation

Check the summary statistics:

```bash
# Look for the summary in logs
grep "Skipped (no summary)" training.log

# Or check the prepared samples
cat ./checkpoints/vlm_finetuned/vlm_data/train/all_samples.json | \
    jq '[.[] | select(.summary != "")] | length'
```

## Example Scenarios

### Scenario 1: All Videos Have Summaries

```
============================================================
VLM Data Preparation Complete
============================================================
Prepared samples: 171
Skipped (no summary): 0
Saved to: ./checkpoints/vlm_finetuned/vlm_data/train/all_samples.json
============================================================
```

Perfect! All videos can be used for training.

### Scenario 2: Some Videos Missing Summaries

```
============================================================
VLM Data Preparation Complete
============================================================
Prepared samples: 165
Skipped (no summary): 6
Saved to: ./checkpoints/vlm_finetuned/vlm_data/train/all_samples.json
============================================================
```

6 videos don't have summaries and are automatically excluded.

### Scenario 3: No Videos Have Summaries

```
============================================================
VLM Data Preparation Complete
============================================================
Prepared samples: 0
Skipped (no summary): 171
Saved to: ./checkpoints/vlm_finetuned/vlm_data/train/all_samples.json
============================================================
```

**Warning!** No training data available. You need to add summaries to your CSV.

## Adding Summaries to CSV

If you have videos without summaries, you can:

1. **Add expert summaries manually** to the CSV
2. **Remove videos** without summaries from the CSV
3. **Generate summaries** using another method (not recommended for medical data)

The CSV should have this structure:

```csv
clip_id,file_path,diagnostic_class,subtype,anatomical_subclass,fps,frame_count,width,height,duration_seconds,summary,diagnosis_text
102769_00036,clips/...,rd,macula_intact,TD,30.0,90,432,432,3.0,"The ocular ultrasound video presents...","<diagnostic>rd</diagnostic>..."
```

## Summary

✓ Videos without summaries are automatically skipped  
✓ Filtering happens at data preparation stage  
✓ Additional safety checks during dataset loading  
✓ Clear reporting of skipped videos  
✓ No training failures or errors  
✓ Only real expert summaries are used  

**Your VLM training is robust and only uses high-quality expert data!**
