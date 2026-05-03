# Bug Fix: Tuple Index Out of Range (Squeeze Dimension Issue)

## Issue

When running `train_llm.py`, the error occurred:
```
Error processing 102769_00036: tuple index out of range
Full traceback:
  File "/content/eyeball-neurips/model/multiclass_model.py", line 228, in extract_important_frames
    top_k = min(top_k, frame_importance.shape[1])
                       ~~~~~~~~~~~~~~~~~~~~~~^^^
IndexError: tuple index out of range
```

## Root Cause

The error was caused by **incorrect use of `.squeeze()`** in the `FrameImportanceModule`:

### Problem Code (multiclass_model.py, line 27)

```python
# WRONG: squeeze() removes ALL dimensions of size 1
return weighted_features, importance_scores.squeeze()
```

### What Was Happening

1. **Input shape**: `importance_scores` is `(B, 1, T, 1, 1)` where B=batch size, T=temporal frames
2. **After `.squeeze()`**: 
   - If B > 1: Result is `(B, T)` ✅ CORRECT
   - **If B = 1**: Result is `(T,)` ❌ WRONG - batch dimension removed!
3. **In `extract_important_frames`**: Code tries to access `frame_importance.shape[1]`
4. **Error**: When B=1, `frame_importance` is 1D tensor `(T,)`, so `.shape[1]` doesn't exist!

### Why This Happened

The VLM data preparation processes videos **one at a time** with batch size 1:
```python
video_tensor = video.unsqueeze(0)  # Add batch dimension → (1, C, T, H, W)
```

When the model processes this single video:
- `importance_scores` becomes `(1, 1, T, 1, 1)`
- `.squeeze()` removes **all** singleton dimensions including the batch dimension
- Result: `(T,)` instead of `(1, T)`

## Solution

### Fix: Squeeze Only Specific Dimensions

```python
# CORRECT: Explicitly squeeze only the dimensions we want to remove
return weighted_features, importance_scores.squeeze(-1).squeeze(-1).squeeze(1)  # (B, T)
```

**Explanation**:
- `.squeeze(-1)`: Remove last dimension (1) → `(B, 1, T, 1)`
- `.squeeze(-1)`: Remove last dimension (1) → `(B, 1, T)`
- `.squeeze(1)`: Remove dimension at index 1 (1) → `(B, T)`

This ensures:
- Batch dimension (index 0) is **always preserved**
- Temporal dimension (index 2 → 1 after squeezes) is **always preserved**
- Result is **always** `(B, T)` regardless of batch size

### Alternative Solutions

**Option 1**: Use specific dimension indices
```python
return weighted_features, importance_scores.squeeze(1).squeeze(-1).squeeze(-1)
```

**Option 2**: Use reshape (most explicit)
```python
B, _, T, _, _ = importance_scores.shape
return weighted_features, importance_scores.reshape(B, T)
```

**Option 3**: Use view (similar to reshape)
```python
return weighted_features, importance_scores.view(importance_scores.size(0), -1)
```

We chose **Option 1** (squeeze specific dims) as it's clear and safe.

## Files Modified

**`multiclass_model.py`** (line 27-28):
```diff
- return weighted_features, importance_scores.squeeze()
+ # Squeeze only the singleton dimensions (1, 1) but keep batch and temporal dims
+ return weighted_features, importance_scores.squeeze(-1).squeeze(-1).squeeze(1)  # (B, T)
```

## Testing

### Before Fix
```python
# Batch size = 1
importance_scores = torch.randn(1, 1, 32, 1, 1)
result = importance_scores.squeeze()
print(result.shape)  # torch.Size([32]) ❌ WRONG - 1D tensor
print(result.shape[1])  # IndexError: tuple index out of range
```

### After Fix
```python
# Batch size = 1
importance_scores = torch.randn(1, 1, 32, 1, 1)
result = importance_scores.squeeze(-1).squeeze(-1).squeeze(1)
print(result.shape)  # torch.Size([1, 32]) ✅ CORRECT - 2D tensor
print(result.shape[1])  # 32 ✅ Works!

# Batch size > 1 (also works)
importance_scores = torch.randn(8, 1, 32, 1, 1)
result = importance_scores.squeeze(-1).squeeze(-1).squeeze(1)
print(result.shape)  # torch.Size([8, 32]) ✅ CORRECT
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
- ✅ No "tuple index out of range" errors
- ✅ `frame_importance` is always `(B, T)` shape
- ✅ `extract_important_frames` works correctly
- ✅ VLM data preparation completes successfully

## Impact on Other Code

This fix ensures that **all code expecting `(B, T)` shaped tensors** will work correctly:

1. **`extract_important_frames`** (line 228):
   ```python
   top_k = min(top_k, frame_importance.shape[1])  # ✅ Now works
   ```

2. **`predict_video`** in `vlm_data_preparation.py`:
   ```python
   frame_importance = attention_dict['frame_importance']  # ✅ Now (B, T)
   ```

3. **Any future code** that processes frame importance scores

## Lessons Learned

1. **Avoid bare `.squeeze()`**: It removes ALL singleton dimensions, which can be unpredictable
2. **Always specify dimensions**: Use `.squeeze(dim)` to be explicit about which dimensions to remove
3. **Test with batch_size=1**: Many bugs only appear when processing single samples
4. **Document tensor shapes**: Add comments like `# (B, T)` to clarify expected shapes
5. **Use assertions in development**: Add `assert frame_importance.dim() == 2` to catch shape issues early

## Related Issues

This is a common PyTorch pitfall:
- `.squeeze()` is convenient but dangerous when batch size varies
- Always prefer `.squeeze(dim)` or `.reshape()` for production code
- Similar issues can occur with `.unsqueeze()` if not careful about dimension indices

## Prevention

To prevent similar issues in the future:

1. **Add shape assertions**:
```python
def forward(self, x):
    # ... processing ...
    result = importance_scores.squeeze(-1).squeeze(-1).squeeze(1)
    assert result.dim() == 2, f"Expected 2D tensor, got {result.dim()}D"
    assert result.shape[0] == x.shape[0], "Batch dimension mismatch"
    return weighted_features, result
```

2. **Add docstring with shapes**:
```python
def forward(self, x):
    """
    Args:
        x: (B, C, T, H, W)
    Returns:
        weighted_features: (B, C, T, H, W)
        importance_scores: (B, T)  # ← Explicit shape documentation
    """
```

3. **Unit tests with different batch sizes**:
```python
def test_frame_importance():
    module = FrameImportanceModule()
    # Test batch_size=1
    x1 = torch.randn(1, 512, 32, 7, 7)
    _, scores1 = module(x1)
    assert scores1.shape == (1, 32)
    
    # Test batch_size=8
    x8 = torch.randn(8, 512, 32, 7, 7)
    _, scores8 = module(x8)
    assert scores8.shape == (8, 32)
```
