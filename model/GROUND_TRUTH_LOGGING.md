# Ground Truth Logging During VLM Finetuning

## Overview

The VLM finetuning code now automatically logs ground truth text samples during training to help you verify what the model is learning.

## What Gets Logged

### 1. During Dataset Loading (First 3 Samples)

When the dataset is first loaded, the first 3 samples will be logged with their complete ground truth:

```
================================================================================
SAMPLE 0 - Video ID: 164267_02030_correct
Is Contrastive: False
================================================================================
GROUND TRUTH TEXT:
--------------------------------------------------------------------------------
The retinal detachment is clearly visible with the characteristic V-shaped 
membrane floating in the vitreous cavity. The macula appears detached, 
indicated by the separation of the retinal layers in the posterior pole...

**Structured Diagnosis:**
<diagnostic>RD</diagnostic>
<subtype>Macula Detached</subtype>
<anatomical>superior</anatomical>
================================================================================
```

### 2. At the Beginning of Each Epoch

At the start of each training epoch, 5 random samples are logged showing:
- Video ID
- Classifier predictions (diagnostic + subtype with confidence)
- Full ground truth text (or truncated if > 500 characters)

Example output:

```
================================================================================
EPOCH 0 - GROUND TRUTH SAMPLES
================================================================================

--------------------------------------------------------------------------------
Sample 1/5 - Video ID: 164267_02030_correct
--------------------------------------------------------------------------------
Predictions:
  - Diagnostic: RD (95.2%)
  - Subtype: Macula Detached (87.5%)

Ground Truth Text:
----------------------------------------
The retinal detachment is clearly visible with the characteristic V-shaped 
membrane floating in the vitreous cavity. The macula appears detached...
[Truncated - full length: 1234 chars]
--------------------------------------------------------------------------------
```

## Where to Find the Logs

### Console Output

All logs are printed to the console during training. You'll see them:
- When training starts (first 3 samples)
- At the beginning of each epoch (5 random samples)

### Log Files

If you're redirecting output to a file:

```bash
bash run_training.sh 2>&1 | tee training.log
```

Then search the log file:

```bash
# Find all ground truth samples
grep -A 20 "GROUND TRUTH TEXT" training.log

# Find epoch summaries
grep -A 50 "EPOCH.*GROUND TRUTH SAMPLES" training.log
```

### TensorBoard

Training metrics are logged to TensorBoard (but not the text samples):

```bash
tensorboard --logdir ./checkpoints/vlm_finetuned
```

## What the Ground Truth Comes From

The ground truth text comes from different sources depending on the sample:

### For Correct Samples (70% of data)

1. **Primary Source**: `summary` column from `balanced_split_desc.csv`
   - Expert-written clinical descriptions
   - Real medical knowledge from your dataset

2. **Structured Diagnosis**: Parsed from CSV columns
   ```
   <diagnostic>RD</diagnostic>
   <subtype>Macula Detached</subtype>
   <anatomical>superior</anatomical>
   ```

3. **Fallback**: Template-generated response if CSV summary is missing

### For Contrastive Samples (30% of data)

Fixed uncertainty response:
```
I notice the highlighted regions in these images appear random or 
inconsistent with typical diagnostic patterns. The heatmap does not 
clearly indicate specific anatomical structures...
```

## Example Training Output

Here's what you'll see when training starts:

```bash
$ bash run_training.sh

==========================================
ERDES Medical Video Diagnosis Training
==========================================

...

[VLM Data Preparation]
Loading samples from: ./checkpoints/vlm_finetuned/vlm_data/vlm_samples.json
Loaded 120 correct samples
Loaded 51 contrastive samples

================================================================================
SAMPLE 0 - Video ID: 164267_02030_correct
Is Contrastive: False
================================================================================
GROUND TRUTH TEXT:
--------------------------------------------------------------------------------
Based on the ultrasound imaging, there is clear evidence of retinal 
detachment. The characteristic V-shaped membrane is visible in the 
vitreous cavity, showing typical undulating motion. The macula appears 
detached, as indicated by the separation of retinal layers in the 
posterior pole. Key diagnostic features include:

1. Detached retinal membrane with characteristic appearance
2. Subretinal space with echo-free appearance
3. Point of attachment near optic nerve head
4. Macular involvement with clear separation from RPE

**Structured Diagnosis:**
<diagnostic>RD</diagnostic>
<subtype>Macula Detached</subtype>
<anatomical>superior</anatomical>
================================================================================

...

[Training Started]

================================================================================
EPOCH 0 - GROUND TRUTH SAMPLES
================================================================================

--------------------------------------------------------------------------------
Sample 1/5 - Video ID: 171234_01234_correct
--------------------------------------------------------------------------------
Predictions:
  - Diagnostic: Non-RD (92.3%)
  - Subtype: PVD (88.1%)

Ground Truth Text:
----------------------------------------
The examination shows posterior vitreous detachment without retinal 
involvement. The vitreous cavity demonstrates characteristic separation 
from the posterior retinal surface. The Weiss ring is visible, indicating 
complete PVD. No retinal detachment is evident...
--------------------------------------------------------------------------------

...

Training: [Epoch 0/10] [Step 10/500] Loss: 2.345
Training: [Epoch 0/10] [Step 20/500] Loss: 2.123
...
```

## Customization

### Change Number of Samples Logged

Edit `vlm_finetuning.py`:

```python
# In train_vlm function, change num_samples parameter
gt_callback = GroundTruthLoggingCallback(train_dataset, num_samples=10)  # Log 10 instead of 5
```

### Log More Initial Samples

Edit `vlm_finetuning.py` in the `__getitem__` method:

```python
# Change from idx < 3 to idx < 10 to log first 10 samples
if idx < 10 and idx not in getattr(self, '_logged_samples', set()):
    ...
```

### Save Ground Truth to File

Add this to your training script:

```python
# After creating the dataset
import json

# Extract all ground truth samples
ground_truths = []
for sample in train_dataset.correct_samples:
    gt = {
        'video_id': sample['video_id'],
        'predictions': sample['predictions'],
        'ground_truth': sample.get('summary', 'N/A')
    }
    ground_truths.append(gt)

# Save to file
with open('./checkpoints/vlm_finetuned/ground_truth_samples.json', 'w') as f:
    json.dump(ground_truths, f, indent=2)

print(f"Saved {len(ground_truths)} ground truth samples")
```

## Verification

Use the logged ground truth to verify:

1. **Data Quality**: Are the expert summaries meaningful?
2. **Alignment**: Do predictions match the ground truth diagnosis?
3. **Coverage**: Are all diagnostic categories represented?
4. **Consistency**: Are similar cases getting similar descriptions?

## Troubleshooting

### No Ground Truth Logged

If you see `[No ground truth available - will use template]`:
- Check that `balanced_split_desc.csv` has a `summary` column
- Verify the CSV is being loaded correctly
- Check that video IDs match between CSV and video files

### Truncated Text

If ground truth is truncated (> 500 chars):
- This is normal for long clinical descriptions
- Full text is still used for training
- Only display is truncated to keep logs readable

### Missing Structured Diagnosis

If you don't see `<diagnostic>` tags:
- Check that CSV has the required columns: `diagnostic_class`, `subtype`, `anatomical_subclass`
- Verify the data preparation step parsed these correctly

## Summary

The ground truth logging helps you:
- ✓ Verify what the model is learning
- ✓ Debug data preparation issues
- ✓ Ensure expert knowledge is being used
- ✓ Monitor training progress qualitatively
- ✓ Validate contrastive learning is working

All logging happens automatically - just run training and check the console output!
