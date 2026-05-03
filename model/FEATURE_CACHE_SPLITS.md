# Feature: Train/Test Split Caching

## Overview

Added caching for train/test splits in `train_llm.py` to avoid regenerating the same splits on every run. The splits are now saved to a JSON file and reused across multiple training sessions.

## Motivation

**Problem**: Creating balanced train/test splits requires:
- Iterating through entire dataset to extract labels
- Computing stratification keys
- Running train_test_split algorithm
- Takes time on every run, even though splits should be deterministic

**Solution**: Cache the split indices and reuse them if parameters haven't changed.

## Implementation

### 1. Enhanced `get_balanced_splits()` Function

Added caching capability with validation:

```python
def get_balanced_splits(dataset, test_size=0.2, random_state=42, cache_file=None):
    """
    Create balanced train/test splits with caching
    
    Args:
        dataset: Full dataset
        test_size: Fraction for test set
        random_state: Random seed
        cache_file: Path to cache file (if None, no caching)
        
    Returns:
        train_indices, test_indices
    """
```

### 2. Cache Validation

The cache is validated before use:
- ✅ Checks if cache file exists
- ✅ Validates `test_size` matches
- ✅ Validates `random_state` matches
- ✅ Validates `dataset_size` matches
- ✅ If any mismatch: regenerates splits

### 3. Automatic Cache Management

**Cache location**: `./checkpoints/vlm/train_test_splits.json`

**Cache structure**:
```json
{
  "test_size": 0.2,
  "random_state": 42,
  "dataset_size": 636,
  "train_indices": [0, 5, 7, 12, ...],
  "test_indices": [1, 3, 8, 15, ...],
  "timestamp": "20260503_061500"
}
```

## Usage

### First Run (No Cache)

```bash
python train_llm.py \
    --classifier_checkpoint ./checkpoints/multiclass/best_model_weights.pth \
    --csv_path ../benchmarks/input/balanced_split_desc.csv \
    --data_root ../erdes \
    --output_dir ./checkpoints/vlm
```

**Output**:
```
Creating balanced train/test splits...
Generating balanced train/test splits...
✓ Saved splits to cache: ./checkpoints/vlm/train_test_splits.json
Train samples: 508
Test samples: 128
```

### Second Run (Cache Exists)

```bash
# Same command
python train_llm.py ...
```

**Output**:
```
Creating balanced train/test splits...
✓ Loaded splits from cache: 508 train, 128 test
Train samples: 508
Test samples: 128
```

### Cache Invalidation (Parameters Changed)

```bash
python train_llm.py ... --test_size 0.3  # Different test_size
```

**Output**:
```
Creating balanced train/test splits...
Cache exists but parameters don't match, regenerating splits...
Generating balanced train/test splits...
✓ Saved splits to cache: ./checkpoints/vlm/train_test_splits.json
Train samples: 445
Test samples: 191
```

## Benefits

### Time Savings
- **First run**: ~2-5 seconds to generate splits
- **Subsequent runs**: ~0.1 seconds to load from cache
- **Saved**: 2-5 seconds per run

### Consistency
- ✅ **Reproducible splits**: Same splits across multiple runs
- ✅ **Deterministic**: Given same parameters, always same splits
- ✅ **Validated**: Cache automatically invalidated if parameters change

### Use Cases

**Scenario 1: Iterative VLM Training**
```bash
# First run - generate splits
python train_llm.py --vlm_lr 1e-5

# Second run - reuse splits ✅
python train_llm.py --vlm_lr 2e-5

# Third run - reuse splits ✅
python train_llm.py --vlm_epochs 20
```

**Scenario 2: Resume After Crash**
```bash
# Training crashes
python train_llm.py ...

# Resume - same splits guaranteed ✅
python train_llm.py ...
```

**Scenario 3: Different Split Ratios**
```bash
# 80/20 split
python train_llm.py --test_size 0.2

# 70/30 split - cache invalidated, new splits generated ✅
python train_llm.py --test_size 0.3
```

## Cache Validation Logic

The cache is considered **valid** if:
1. ✅ Cache file exists
2. ✅ `test_size` matches current setting
3. ✅ `random_state` matches current setting
4. ✅ `dataset_size` matches current dataset

The cache is **invalidated** if:
1. ❌ Cache file doesn't exist
2. ❌ `test_size` changed
3. ❌ `random_state` changed
4. ❌ `dataset_size` changed (dataset updated)
5. ❌ Cache file is corrupted

## Implementation Details

### Files Modified

**`train_llm.py`**:
1. Updated `get_balanced_splits()` function (lines 30-103):
   - Added `cache_file` parameter
   - Added cache loading logic
   - Added cache validation
   - Added cache saving logic
   
2. Updated `main()` function (lines 312-320):
   - Added `split_cache_file` path
   - Passed cache file to `get_balanced_splits()`

### Key Code Changes

**Cache loading**:
```python
# Try to load from cache
if cache_file and Path(cache_file).exists():
    try:
        with open(cache_file, 'r') as f:
            cache_data = json.load(f)
        
        # Validate cache matches current settings
        if (cache_data.get('test_size') == test_size and 
            cache_data.get('random_state') == random_state and
            cache_data.get('dataset_size') == len(dataset)):
            
            train_indices = cache_data['train_indices']
            test_indices = cache_data['test_indices']
            print(f"✓ Loaded splits from cache: {len(train_indices)} train, {len(test_indices)} test")
            return train_indices, test_indices
```

**Cache saving**:
```python
# Save to cache
if cache_file:
    cache_data = {
        'test_size': test_size,
        'random_state': random_state,
        'dataset_size': len(dataset),
        'train_indices': train_indices,
        'test_indices': test_indices,
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
    }
    
    Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, 'w') as f:
        json.dump(cache_data, f, indent=2)
    print(f"✓ Saved splits to cache: {cache_file}")
```

## Testing

### Test Case 1: No Cache
```bash
rm -f ./checkpoints/vlm/train_test_splits.json
python train_llm.py ...
# Expected: Generates and saves splits
```

### Test Case 2: Valid Cache
```bash
python train_llm.py ...  # Run twice
# Expected: Second run loads from cache
```

### Test Case 3: Changed test_size
```bash
python train_llm.py --test_size 0.2  # First run
python train_llm.py --test_size 0.3  # Second run
# Expected: Regenerates splits, updates cache
```

### Test Case 4: Changed random_state
```bash
python train_llm.py --random_state 42  # First run
python train_llm.py --random_state 123  # Second run
# Expected: Regenerates splits, updates cache
```

### Test Case 5: Corrupted Cache
```bash
echo "corrupted" > ./checkpoints/vlm/train_test_splits.json
python train_llm.py ...
# Expected: Detects corruption, regenerates splits
```

## Cache Location

```
./checkpoints/vlm/
├── train_test_splits.json  ← Split indices cache
├── vlm_data/
│   ├── train/
│   │   └── all_samples.json
│   └── test/
│       └── all_samples.json
└── config.json
```

## Interaction with Other Caches

This feature works alongside the VLM data cache:

1. **Split cache**: Stores train/test indices
   - Fast to load (~0.1s)
   - Small file size (~few KB)
   - Invalidated if parameters change

2. **VLM data cache**: Stores prepared frames/heatmaps
   - Slower to generate (~10-15 min)
   - Large file size (~GB)
   - Uses split indices to organize data

Both caches work together for maximum efficiency!

## Future Enhancements

Potential improvements:
1. **Multiple cache versions**: Keep caches for different parameter combinations
2. **Cache metadata**: Store more info (dataset hash, creation time, etc.)
3. **Cache cleanup**: Automatically remove old/unused caches
4. **Visual diff**: Show what changed when cache is invalidated

## Backward Compatibility

✅ **Fully backward compatible**:
- Default behavior unchanged (auto-generates splits)
- If `cache_file=None`, works exactly as before
- No breaking changes to function signature
- Old code continues to work

## Summary

This feature adds intelligent caching to train/test split generation, ensuring:
- ✅ **Fast**: Instant loading from cache
- ✅ **Consistent**: Same splits across runs
- ✅ **Safe**: Automatic validation and invalidation
- ✅ **Transparent**: Clear messages about cache status
- ✅ **Robust**: Handles corruption and parameter changes gracefully

Combined with VLM data caching, this significantly speeds up iterative experimentation!
