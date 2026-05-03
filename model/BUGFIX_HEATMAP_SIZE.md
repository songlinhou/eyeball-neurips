# Bug Fix: OpenCV Size Mismatch in Heatmap Overlay

## Issue

When running `train_llm.py`, the following error occurred:
```
Error processing 164267_01174: OpenCV(4.13.0) /io/opencv/modules/core/src/arithm.cpp:662: error: (-209:Sizes of input arguments do not match)
...
cv2.error: ... Sizes of input arguments do not match
```

## Root Cause

The error occurred in `generate_heatmap_overlay()` when trying to blend the frame and heatmap using `cv2.addWeighted()`:

### Problem

```python
# Line 136 in vlm_data_preparation.py
overlay = cv2.addWeighted(frame, 1 - alpha, heatmap, alpha, 0)
```

**Why it failed**:
1. `frame` has shape `(224, 224, 3)` - full resolution RGB image
2. `attention_map` has shape `(7, 7)` or `(14, 14)` - low resolution from model's spatial attention
3. After applying colormap, `heatmap` has shape `(7, 7, 3)` or `(14, 14, 3)`
4. `cv2.addWeighted()` requires both inputs to have **exactly the same shape**
5. **Mismatch**: `(224, 224, 3)` ≠ `(7, 7, 3)` → Error!

### Why This Happens

The spatial attention map comes from the model's intermediate layers (after pooling/downsampling):
- Input video: `(B, C, T, 224, 224)`
- After ResNet layers: `(B, 512, T', 7, 7)` or `(B, 512, T', 14, 14)`
- Spatial attention: `(B, 1, T', 7, 7)` or `(B, 1, T', 14, 14)`

The attention map has lower spatial resolution than the original frame!

## Solution

### Fix: Resize Attention Map to Match Frame Size

```python
def generate_heatmap_overlay(self, frame: np.ndarray, attention_map: np.ndarray, 
                             alpha=0.5, colormap='jet') -> np.ndarray:
    # Ensure frame is uint8
    if frame.max() <= 1.0:
        frame = (frame * 255).astype(np.uint8)
    else:
        frame = frame.astype(np.uint8)
    
    # Get frame dimensions
    H, W = frame.shape[:2]
    
    # ✅ NEW: Resize attention map to match frame size if needed
    if attention_map.shape != (H, W):
        attention_map = cv2.resize(attention_map, (W, H), interpolation=cv2.INTER_LINEAR)
    
    # Normalize attention map
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
    
    # Apply colormap
    cmap = cm.get_cmap(colormap)
    heatmap = cmap(attention_map)[:, :, :3]  # Remove alpha channel
    heatmap = (heatmap * 255).astype(np.uint8)
    
    # Blend (now both have same size!)
    overlay = cv2.addWeighted(frame, 1 - alpha, heatmap, alpha, 0)
    
    return overlay
```

### Key Changes

1. **Added size check** (line 131-132):
   ```python
   if attention_map.shape != (H, W):
       attention_map = cv2.resize(attention_map, (W, H), interpolation=cv2.INTER_LINEAR)
   ```

2. **Moved frame conversion earlier**: Process frame to uint8 before getting dimensions

3. **Used INTER_LINEAR interpolation**: Smooth upsampling for attention maps

## Files Modified

**`vlm_data_preparation.py`** (lines 107-145):
- Added attention map resizing to match frame dimensions
- Reordered operations for clarity
- Updated docstring to reflect that attention_map can have different size

## Testing

### Before Fix
```python
frame = np.random.rand(224, 224, 3) * 255  # (224, 224, 3)
attention = np.random.rand(7, 7)            # (7, 7)
overlay = generate_heatmap_overlay(frame, attention)
# ❌ Error: Sizes of input arguments do not match
```

### After Fix
```python
frame = np.random.rand(224, 224, 3) * 255  # (224, 224, 3)
attention = np.random.rand(7, 7)            # (7, 7)
overlay = generate_heatmap_overlay(frame, attention)
# ✅ Works! attention resized to (224, 224) internally
print(overlay.shape)  # (224, 224, 3)
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
- ✅ No OpenCV size mismatch errors
- ✅ Heatmap overlays generated correctly
- ✅ Attention maps upsampled smoothly to frame resolution
- ✅ VLM data preparation completes successfully

## Impact

This fix ensures that:
1. **Attention maps at any resolution** can be overlaid on frames
2. **Smooth upsampling** preserves attention patterns while matching frame size
3. **No manual resizing needed** - handled automatically in the function

## Visual Quality

Using `cv2.INTER_LINEAR` for upsampling:
- ✅ Smooth gradients in attention maps
- ✅ No blocky artifacts
- ✅ Preserves relative importance scores
- ✅ Suitable for visualization

Alternative interpolation methods:
- `INTER_NEAREST`: Faster but blocky (not recommended for visualization)
- `INTER_CUBIC`: Smoother but slower (overkill for attention maps)
- `INTER_LINEAR`: Good balance of speed and quality ✅

## Lessons Learned

1. **Always check tensor/array dimensions** before operations that require matching sizes
2. **Document expected shapes** in docstrings (e.g., `(H_attn, W_attn)` vs `(H, W)`)
3. **Handle resolution mismatches gracefully** - don't assume inputs have the same size
4. **Use appropriate interpolation** - linear for smooth heatmaps, nearest for masks
5. **Test with different model architectures** - attention map sizes vary by layer depth

## Related Considerations

### Why Not Resize Frame Instead?

We could resize the frame down to match attention map size:
```python
# Alternative (not recommended)
frame_small = cv2.resize(frame, (attention_map.shape[1], attention_map.shape[0]))
overlay_small = cv2.addWeighted(frame_small, 1-alpha, heatmap, alpha, 0)
overlay = cv2.resize(overlay_small, (W, H))
```

**Why we don't do this**:
- ❌ Loses frame detail
- ❌ Two resize operations instead of one
- ❌ Attention map is already low-res, no benefit to downsizing frame

### Why Resize Before Colormap?

We resize the attention map **before** applying the colormap:
```python
# ✅ CORRECT: Resize first, then colormap
attention_map = cv2.resize(attention_map, (W, H))
heatmap = cmap(attention_map)[:, :, :3]
```

Not after:
```python
# ❌ WRONG: Colormap first, then resize
heatmap = cmap(attention_map)[:, :, :3]
heatmap = cv2.resize(heatmap, (W, H))
```

**Reason**: Resizing grayscale attention map is simpler and preserves values better than resizing RGB heatmap.
