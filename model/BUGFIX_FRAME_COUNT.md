# Bug Fix: Index Out of Bounds for Frame Count

## Issue

When running `train_llm.py`, the following error occurred:
```
Error processing 102769_00036: index 4 is out of bounds for axis 0 with size 4
Full traceback:
  File "/content/eyeball-neurips/model/vlm_data_preparation.py", line 233, in prepare_vlm_sample
    frame = important_frames[k].transpose(1, 2, 0)
            ~~~~~~~~~~~~~~~~^^^
IndexError: index 4 is out of bounds for axis 0 with size 4
```

## Root Cause

The error occurred because the code was trying to access more frames than were actually extracted:

### Problem Code

```python
# Line 231 in vlm_data_preparation.py
for k in range(self.top_k_frames):  # self.top_k_frames = 5
    frame = important_frames[k].transpose(1, 2, 0)  # ❌ important_frames only has 4 elements
```

### What Was Happening

1. **Configuration**: `self.top_k_frames = 5` (we want top 5 frames)
2. **Video has limited frames**: Some videos have fewer than 5 temporal frames after downsampling
3. **Model adjusts**: In `multiclass_model.py`, line 228:
   ```python
   top_k = min(top_k, frame_importance.shape[1])  # Adjusts to actual frame count
   ```
4. **Result**: `important_frames` array has only 4 elements (indices 0-3)
5. **Loop tries to access**: `range(5)` → tries to access indices 0, 1, 2, 3, **4**
6. **Error**: Index 4 doesn't exist in array of size 4!

### Why This Happens

Videos can have different numbers of frames:
- **Short videos**: May have only 10-15 frames total
- **After sampling**: Dataset samples 32 frames uniformly
- **After model processing**: ResNet downsamples temporal dimension
- **Final temporal frames**: May be 4, 8, 16, etc. depending on video length and model architecture

The model's `extract_important_frames` correctly handles this by adjusting `top_k`, but the VLM data preparation code was using the **requested** number instead of the **actual** number extracted.

## Solution

### Fix: Use Actual Number of Frames Extracted

**In `prepare_vlm_sample` method** (lines 231-234):
```python
# ✅ CORRECT: Use actual number of frames
num_frames = len(important_frames)

for k in range(num_frames):  # Iterate only over available frames
    frame = important_frames[k].transpose(1, 2, 0)
```

**In `create_contrastive_samples` method** (lines 332-335):
```python
# ✅ CORRECT: Use actual number of frames
num_frames = len(important_frames)

for k in range(num_frames):
    frame = important_frames[k].transpose(1, 2, 0)
```

### Why This Works

- `len(important_frames)` returns the **actual** number of frames extracted
- If video has 4 frames: `num_frames = 4`, loop runs 0-3 ✅
- If video has 5+ frames: `num_frames = 5`, loop runs 0-4 ✅
- No index out of bounds errors!

## Files Modified

**`vlm_data_preparation.py`**:

1. **Lines 231-234** (`prepare_vlm_sample`):
   ```diff
   + # Use actual number of frames extracted (may be less than top_k_frames)
   + num_frames = len(important_frames)
   + 
   - for k in range(self.top_k_frames):
   + for k in range(num_frames):
   ```

2. **Lines 332-335** (`create_contrastive_samples`):
   ```diff
   + # Use actual number of frames extracted (may be less than top_k_frames)
   + num_frames = len(important_frames)
   + 
   - for k in range(self.top_k_frames):
   + for k in range(num_frames):
   ```

## Testing

### Before Fix
```python
top_k_frames = 5
important_frames = np.random.rand(4, 3, 224, 224)  # Only 4 frames

for k in range(top_k_frames):  # Tries to access 0, 1, 2, 3, 4
    frame = important_frames[k]  # ❌ IndexError at k=4
```

### After Fix
```python
top_k_frames = 5
important_frames = np.random.rand(4, 3, 224, 224)  # Only 4 frames
num_frames = len(important_frames)  # num_frames = 4

for k in range(num_frames):  # Only accesses 0, 1, 2, 3
    frame = important_frames[k]  # ✅ Works!
```

## Verification

Run the training script:
```bash
python train_llm.py \
    --classifier_checkpoint ./checkpoints/multiclass/best_model_weights.pth \
    --csv_path ../benchmarks/input/balanced_split_desc.csv \
    --data_root ../erdes \
    --output_dir ./checkpoints/vlm \
    --use_contrastive
```

Expected behavior:
- ✅ No index out of bounds errors
- ✅ Works with videos of any length
- ✅ Adapts to actual number of frames extracted
- ✅ VLM samples may have 1-5 frames depending on video

## Impact

This fix ensures that:
1. **Short videos work correctly**: Videos with <5 frames after processing don't cause errors
2. **Flexible frame count**: Each VLM sample has the appropriate number of frames
3. **No data loss**: All available important frames are used, even if fewer than requested
4. **Consistent behavior**: Aligns with model's `extract_important_frames` logic

## Sample Counts

After this fix, VLM samples will have varying numbers of frames:

| Video Length | Frames After Model | Frames in VLM Sample |
|--------------|-------------------|---------------------|
| Very short   | 2-3 frames        | 2-3 frames          |
| Short        | 4 frames          | 4 frames            |
| Normal       | 5+ frames         | 5 frames (max)      |
| Long         | 8+ frames         | 5 frames (max)      |

This is **expected and correct** - we use as many frames as available, up to the maximum of 5.

## Lessons Learned

1. **Don't assume fixed sizes**: Videos can have varying numbers of frames
2. **Use actual lengths**: Always use `len()` or actual array size, not configuration values
3. **Align with upstream logic**: If model adjusts `top_k`, downstream code should too
4. **Test edge cases**: Test with very short videos to catch these issues
5. **Document assumptions**: Add comments about variable frame counts

## Prevention

To prevent similar issues:

1. **Add assertions**:
```python
num_frames = len(important_frames)
assert num_frames > 0, f"No frames extracted for video {video_id}"
assert num_frames <= self.top_k_frames, f"Got more frames than requested"
```

2. **Log warnings for short videos**:
```python
if num_frames < self.top_k_frames:
    print(f"Warning: Video {video_id} has only {num_frames} frames (requested {self.top_k_frames})")
```

3. **Document in docstring**:
```python
def prepare_vlm_sample(...):
    """
    ...
    Note: The number of frames in the output may be less than top_k_frames
    if the video has fewer temporal frames after model processing.
    """
```

## Related Code

The model's `extract_important_frames` already handles this correctly:
```python
# multiclass_model.py, line 228
top_k = min(top_k, frame_importance.shape[1])  # ✅ Adjusts to available frames
```

Our fix ensures the VLM data preparation code respects this adjustment.
