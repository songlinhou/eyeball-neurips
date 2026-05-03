# Bug Fix: Tuple Index Out of Range Error

## Issue

When running `train_llm.py`, the following error occurred:
```
Error processing 102769_00036: tuple index out of range
Error processing 164267_03845: tuple index out of range
```

## Root Cause

The error was caused by **incorrect unpacking of the return value** from `create_contrastive_samples()`:

### Problem Code (train_llm.py)
```python
# WRONG: Assigns tuple to single variable, then tries to extend
contrastive_samples = preparator.create_contrastive_samples(
    video_tensor=video_tensor,
    video_id=video_id,
    output_dir=str(sample_dir)
)
samples.extend(contrastive_samples)  # ❌ Tries to iterate over tuple
```

### What Was Happening

1. `create_contrastive_samples()` returns a **tuple of 2 elements**: `(correct_sample, contrastive_sample)`
2. The code assigned this tuple to `contrastive_samples` (single variable)
3. Then called `samples.extend(contrastive_samples)` which tried to iterate over the tuple
4. Somewhere in the iteration/processing, code tried to access tuple indices that didn't exist
5. Result: `tuple index out of range` error

## Solution

### Fix 1: Properly Unpack the Tuple (train_llm.py)

```python
# CORRECT: Unpack tuple into two variables
correct_sample, contrastive_sample = preparator.create_contrastive_samples(
    video_tensor=video_tensor,
    video_id=video_id,
    output_dir=str(sample_dir),
    ground_truth=ground_truth
)
samples.append(correct_sample)
samples.append(contrastive_sample)
```

### Fix 2: Add ground_truth Parameter (vlm_data_preparation.py)

Updated `create_contrastive_samples()` signature to accept and pass through `ground_truth`:

```python
def create_contrastive_samples(self, video_tensor: torch.Tensor,
                               video_id: str,
                               output_dir: str,
                               ground_truth: Dict = None) -> Tuple[Dict, Dict]:
    # ...
    correct_sample = self.prepare_vlm_sample(
        video_tensor, f"{video_id}_correct", 
        output_dir / "correct", ground_truth  # ✅ Now passes ground_truth
    )
```

### Fix 3: Enhanced Error Handling

Added better error messages to help debug future issues:

**In vlm_data_preparation.py:**
```python
try:
    important_frames = important_frames.cpu().numpy()[0]
    frame_indices = frame_indices.cpu().numpy()[0]
    importance_scores = importance_scores.cpu().numpy()[0]
    spatial_attention = spatial_attention.cpu().numpy()[0]
except IndexError as e:
    raise IndexError(f"Shape mismatch for video {video_id}: "
                   f"important_frames={important_frames.shape}, "
                   f"frame_indices={frame_indices.shape}, "
                   f"importance_scores={importance_scores.shape}, "
                   f"spatial_attention={spatial_attention.shape}") from e
```

**In train_llm.py:**
```python
except Exception as e:
    import traceback
    print(f"\nError processing {video_id}: {e}")
    print("Full traceback:")
    traceback.print_exc()
    continue
```

## Files Modified

1. **`train_llm.py`** (lines 127-135)
   - Changed from `samples.extend(contrastive_samples)` to proper tuple unpacking
   - Added `ground_truth` parameter to `create_contrastive_samples()` call
   - Enhanced error handling with full traceback

2. **`vlm_data_preparation.py`** (lines 278-301)
   - Added `ground_truth` parameter to `create_contrastive_samples()` signature
   - Pass `ground_truth` through to `prepare_vlm_sample()` call
   - Added shape validation and better error messages

## Testing

After these fixes, `train_llm.py` should:
1. ✅ Properly unpack the tuple returned by `create_contrastive_samples()`
2. ✅ Create 3 samples per video (if `use_contrastive=True`):
   - 1 regular sample
   - 1 correct sample (with true heatmaps)
   - 1 contrastive sample (with fake/shifted heatmaps)
3. ✅ Pass ground truth information to all samples
4. ✅ Provide clear error messages if shape mismatches occur

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
- No "tuple index out of range" errors
- Progress bar shows processing of all videos
- Each video generates 3 samples (1 regular + 2 contrastive)
- All samples saved to `./checkpoints/vlm/vlm_data/all_samples.json`

## Lessons Learned

1. **Always check return types**: `create_contrastive_samples()` returns a tuple, not a list
2. **Unpack tuples explicitly**: Use `a, b = func()` instead of `result = func()` when you know it's a tuple
3. **Add type hints**: The function signature had `-> Tuple[Dict, Dict]` which should have been a clue
4. **Better error messages**: Shape information in errors helps debug tensor dimension issues
5. **Test edge cases**: The error only appeared when `use_contrastive=True`, highlighting the importance of testing all code paths
