# Bug Fix: Parameter Name Correction

## Issue

Training scripts failed with error:
```
TypeError: ERDESDataset.__init__() got an unexpected keyword argument 'frame_size'
```

## Root Cause

The `ERDESDataset` class uses parameter name `img_size`, but the training scripts were using `frame_size`.

## Files Fixed

### 1. `train_multiclass.py`
- Changed `frame_size` → `img_size` in dataset initialization
- Changed `--frame_size` → `--img_size` in argparse

### 2. `train_llm.py`
- Changed `frame_size` → `img_size` in dataset initialization
- Changed `--frame_size` → `--img_size` in argparse

### 3. `run_training.sh`
- Changed `--frame_size 224` → `--img_size 224` (2 occurrences)

## Correct Usage

```bash
# Correct parameter name
python train_multiclass.py --img_size 224

# NOT frame_size
python train_multiclass.py --frame_size 224  # ❌ WRONG
```

## Status

✅ **FIXED** - All scripts now use the correct parameter name `img_size`

You can now run:
```bash
bash run_training.sh
```

Or:
```bash
python train_multiclass.py --img_size 224
python train_llm.py --img_size 224
```
