# Feature: Skip Classifier Training if Checkpoint Exists

## Overview

Updated `run_training.sh` to automatically skip multi-class classifier training if a checkpoint already exists at the fixed location. This saves significant time when iterating on VLM training.

## Motivation

**Problem**: 
- Classifier training takes ~37 minutes
- When experimenting with VLM settings, the classifier doesn't need to be retrained
- Previous workflow always retrained the classifier, wasting time

**Solution**: 
- Check for existing checkpoint before training
- Use fixed checkpoint path for consistency
- Only train if checkpoint doesn't exist

## Implementation

### Fixed Checkpoint Path

```bash
MULTICLASS_CHECKPOINT="./checkpoints/multiclass/best_model_weights.pth"
```

Instead of timestamped directories, we now use a fixed path for the "production" checkpoint.

### Automatic Skip Logic

```bash
# Check if classifier checkpoint already exists
if [ -f "$MULTICLASS_CHECKPOINT" ]; then
    echo "✓ Found existing classifier checkpoint: $MULTICLASS_CHECKPOINT"
    echo "Skipping classifier training (checkpoint already exists)"
    echo ""
    echo "To retrain classifier, delete or rename:"
    echo "  $MULTICLASS_CHECKPOINT"
    MULTICLASS_OUTPUT="./checkpoints/multiclass"
else
    echo "No existing checkpoint found, training classifier..."
    
    # Create timestamped output directory
    MULTICLASS_OUTPUT="./checkpoints/multiclass_$(date +%Y%m%d_%H%M%S)"
    
    # Train classifier...
    python train_multiclass.py ...
    
    # Copy to fixed location for future runs
    mkdir -p "./checkpoints/multiclass"
    cp "$MULTICLASS_OUTPUT/best_model_weights.pth" "$MULTICLASS_CHECKPOINT"
    echo "✓ Copied checkpoint to: $MULTICLASS_CHECKPOINT"
fi
```

## Usage Examples

### First Run (No Checkpoint)

```bash
bash run_training.sh
```

**Output**:
```
==========================================
Stage 1: Multi-Class Classifier
==========================================
No existing checkpoint found, training classifier...

Output: ./checkpoints/multiclass_20260503_134500

Training classifier...
[... 37 minutes of training ...]

✓ Classifier training complete!
Best model: ./checkpoints/multiclass_20260503_134500/best_model_weights.pth
✓ Copied checkpoint to: ./checkpoints/multiclass/best_model_weights.pth

==========================================
Stage 2 & 3: VLM Data Preparation & Training
==========================================
Using classifier: ./checkpoints/multiclass/best_model_weights.pth
...
```

### Second Run (Checkpoint Exists)

```bash
bash run_training.sh
```

**Output**:
```
==========================================
Stage 1: Multi-Class Classifier
==========================================
✓ Found existing classifier checkpoint: ./checkpoints/multiclass/best_model_weights.pth
Skipping classifier training (checkpoint already exists)

To retrain classifier, delete or rename:
  ./checkpoints/multiclass/best_model_weights.pth

==========================================
Stage 2 & 3: VLM Data Preparation & Training
==========================================
Using classifier: ./checkpoints/multiclass/best_model_weights.pth
...
```

**Time saved**: ~37 minutes! ⚡

### Force Retrain Classifier

If you want to retrain the classifier (e.g., with different hyperparameters):

```bash
# Option 1: Delete the checkpoint
rm ./checkpoints/multiclass/best_model_weights.pth
bash run_training.sh

# Option 2: Rename the checkpoint
mv ./checkpoints/multiclass/best_model_weights.pth \
   ./checkpoints/multiclass/best_model_weights_old.pth
bash run_training.sh
```

### Train Classifier Only

```bash
bash run_training.sh --no-vlm
```

This will train (or skip) the classifier and exit before VLM training.

## Directory Structure

### Before (Timestamped Directories)

```
./checkpoints/
├── multiclass_20260503_120000/
│   └── best_model_weights.pth
├── multiclass_20260503_130000/
│   └── best_model_weights.pth
└── multiclass_20260503_140000/
    └── best_model_weights.pth
```

**Problem**: Hard to know which checkpoint to use, cluttered directories

### After (Fixed Path + Timestamped Archives)

```
./checkpoints/
├── multiclass/
│   └── best_model_weights.pth  ← Fixed "production" checkpoint
├── multiclass_20260503_120000/  ← Timestamped archive (if retrained)
│   ├── best_model_weights.pth
│   ├── best_diagnostic_report.txt
│   ├── best_subtype_report.txt
│   └── history.json
└── vlm_finetuned/
    ├── vlm_data/
    ├── vlm_checkpoints/
    └── train_test_splits.json
```

**Benefits**: 
- Clear "production" checkpoint at fixed path
- Timestamped archives preserved for comparison
- VLM always uses the fixed checkpoint path

## Workflow

### Typical Development Workflow

1. **First time**: Train classifier (~37 min)
   ```bash
   bash run_training.sh --no-vlm
   ```

2. **Iterate on VLM**: Skip classifier training
   ```bash
   bash run_training.sh  # Skips classifier, trains VLM
   bash run_training.sh --resume  # Resume VLM from checkpoint
   ```

3. **Improve classifier**: Retrain when needed
   ```bash
   rm ./checkpoints/multiclass/best_model_weights.pth
   bash run_training.sh --no-vlm
   ```

4. **Full pipeline**: Train both (or skip classifier if exists)
   ```bash
   bash run_training.sh
   ```

## Benefits

### Time Savings

**Scenario: Experimenting with VLM hyperparameters**

Without this feature:
```
Run 1: Classifier (37 min) + VLM (60 min) = 97 min
Run 2: Classifier (37 min) + VLM (60 min) = 97 min
Run 3: Classifier (37 min) + VLM (60 min) = 97 min
Total: 291 minutes (4.85 hours)
```

With this feature:
```
Run 1: Classifier (37 min) + VLM (60 min) = 97 min
Run 2: VLM only (60 min) = 60 min  ← Saved 37 min
Run 3: VLM only (60 min) = 60 min  ← Saved 37 min
Total: 217 minutes (3.62 hours)
```

**Time saved**: 74 minutes (1.23 hours) for 3 runs!

### Consistency

- ✅ **Fixed checkpoint path**: Always use the same checkpoint
- ✅ **No confusion**: Clear which checkpoint is "production"
- ✅ **Reproducible**: VLM experiments use same classifier

### Flexibility

- ✅ **Easy to retrain**: Just delete the checkpoint
- ✅ **Archives preserved**: Timestamped directories kept
- ✅ **Explicit control**: Clear messages about what's happening

## Command-Line Options

### Full Training Pipeline
```bash
bash run_training.sh
```
- Trains classifier (if needed)
- Prepares VLM data
- Trains VLM

### Resume VLM Training
```bash
bash run_training.sh --resume
```
- Skips classifier (if exists)
- Skips VLM data prep (uses cache)
- Resumes VLM from last checkpoint

### Classifier Only
```bash
bash run_training.sh --no-vlm
```
- Trains classifier (if needed)
- Exits before VLM training

### Combined Options
```bash
bash run_training.sh --resume --no-vlm
```
- Skips classifier (if exists)
- Exits (no VLM training)
- Useful for checking if classifier exists

## Implementation Details

### Files Modified

**`run_training.sh`**:
1. Changed `MULTICLASS_OUTPUT` to `MULTICLASS_CHECKPOINT` (fixed path)
2. Added checkpoint existence check (lines 83-135)
3. Copy trained checkpoint to fixed location (lines 130-134)
4. Updated VLM training to use fixed checkpoint path (line 166)
5. Updated final summary to show correct paths (lines 204-210)

### Key Code Changes

**Checkpoint check**:
```bash
if [ -f "$MULTICLASS_CHECKPOINT" ]; then
    echo "✓ Found existing classifier checkpoint"
    echo "Skipping classifier training (checkpoint already exists)"
    MULTICLASS_OUTPUT="./checkpoints/multiclass"
else
    # Train classifier...
fi
```

**Copy to fixed location**:
```bash
# After successful training
mkdir -p "./checkpoints/multiclass"
cp "$MULTICLASS_OUTPUT/best_model_weights.pth" "$MULTICLASS_CHECKPOINT"
echo "✓ Copied checkpoint to: $MULTICLASS_CHECKPOINT"
```

**Use fixed checkpoint for VLM**:
```bash
VLM_CMD="python train_llm.py \
    --classifier_checkpoint $MULTICLASS_CHECKPOINT \
    ..."
```

## Testing

### Test Case 1: No Checkpoint
```bash
rm -f ./checkpoints/multiclass/best_model_weights.pth
bash run_training.sh --no-vlm
# Expected: Trains classifier, copies to fixed location
```

### Test Case 2: Checkpoint Exists
```bash
bash run_training.sh --no-vlm
# Expected: Skips training, uses existing checkpoint
```

### Test Case 3: Full Pipeline with Existing Classifier
```bash
bash run_training.sh
# Expected: Skips classifier, trains VLM
```

### Test Case 4: Force Retrain
```bash
rm ./checkpoints/multiclass/best_model_weights.pth
bash run_training.sh
# Expected: Trains classifier, then VLM
```

## Future Enhancements

Potential improvements:
1. **Checkpoint validation**: Verify checkpoint is valid before using
2. **Version tracking**: Store model architecture version in checkpoint
3. **Automatic retraining**: Retrain if dataset changed
4. **Multiple checkpoints**: Support different classifier variants
5. **Checkpoint comparison**: Compare metrics of different checkpoints

## Backward Compatibility

✅ **Fully backward compatible**:
- Existing timestamped directories still work
- Old checkpoints can be copied to fixed location
- No breaking changes to training scripts
- Clear migration path

## Summary

This feature adds intelligent checkpoint management to the training pipeline:
- ✅ **Automatic**: Skips training if checkpoint exists
- ✅ **Fast**: Saves ~37 minutes per run
- ✅ **Consistent**: Fixed checkpoint path for reproducibility
- ✅ **Flexible**: Easy to retrain when needed
- ✅ **Clear**: Explicit messages about what's happening
- ✅ **Efficient**: Ideal for iterative VLM experimentation

Perfect for rapid iteration on VLM training while keeping classifier consistent! 🚀
