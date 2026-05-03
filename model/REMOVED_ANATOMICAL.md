# Removed Anatomical Subclass from Training

## Summary

Removed `anatomical_subclass` from model training and prediction. The model now only predicts:
- **diagnostic_class**: `non_rd` or `rd` (2 classes)
- **subtype**: `normal`, `macula_intact`, `macula_detached`, `pvd` (4 classes)

The `anatomical_subclass` field is still preserved in metadata for reference but is NOT used for training or prediction.

## Changes Made

### 1. ✅ Model Architecture (`multiclass_model.py`)

**Removed**:
- `num_anatomical_classes` parameter
- `anatomical_classifier` head
- `anatomical` output from forward pass

**Before**:
```python
def __init__(self, 
             num_diagnostic_classes=2,
             num_subtype_classes=4,
             num_anatomical_classes=6,  # REMOVED
             pretrained=True, 
             dropout=0.3):
    ...
    self.anatomical_classifier = nn.Linear(...)  # REMOVED
    
outputs = {
    'diagnostic': ...,
    'subtype': ...,
    'anatomical': ...  # REMOVED
}
```

**After**:
```python
def __init__(self, 
             num_diagnostic_classes=2,
             num_subtype_classes=4,
             pretrained=True, 
             dropout=0.3):
    ...
    # Only diagnostic and subtype classifiers
    
outputs = {
    'diagnostic': ...,
    'subtype': ...
}
```

### 2. ✅ Data Preparation (`vlm_data_preparation.py`)

**Removed**:
- `self.anatomical_labels` mapping
- Anatomical predictions from `predict_video()`
- Anatomical information from prompts
- Anatomical confidence from sample metadata

**Before**:
```python
self.anatomical_labels = {
    0: "N/A",
    1: "TD",
    ...
}

predictions = {
    'diagnostic': {...},
    'subtype': {...},
    'anatomical': {...}  # REMOVED
}

prompt = f"""
Primary Diagnosis: {diagnostic}
Subtype: {subtype}
Anatomical Location: {anatomical}  # REMOVED
"""
```

**After**:
```python
# No anatomical_labels

predictions = {
    'diagnostic': {...},
    'subtype': {...}
}

prompt = f"""
Primary Diagnosis: {diagnostic}
Subtype: {subtype}
"""
```

### 3. ✅ Dataset Loader (`erdes_dataset.py`)

**Changed**:
- Removed `anatomical` from labels dict
- Kept `anatomical_subclass` in metadata for reference only
- Updated collate function to not include anatomical labels

**Before**:
```python
labels = {
    'diagnostic': diagnostic_label,
    'subtype': subtype_label,
    'anatomical': anatomical_label  # REMOVED
}
```

**After**:
```python
labels = {
    'diagnostic': diagnostic_label,
    'subtype': subtype_label
}

metadata = {
    ...
    'anatomical_subclass': anatomical_str,  # For reference only
    ...
}
```

### 4. ✅ Pipeline (`vlm_pipeline.py`)

**Removed**:
- `num_anatomical_classes` parameter
- Anatomical output from diagnosis display

**Before**:
```python
pipeline = VLMDiagnosisPipeline(
    classifier_checkpoint='...',
    num_diagnostic_classes=2,
    num_subtype_classes=4,
    num_anatomical_classes=6  # REMOVED
)

print(f"Anatomical: {sample['predictions']['anatomical']}")  # REMOVED
```

**After**:
```python
pipeline = VLMDiagnosisPipeline(
    classifier_checkpoint='...',
    num_diagnostic_classes=2,
    num_subtype_classes=4
)

# Only diagnostic and subtype printed
```

## Model Structure

### Current (2-Head Model)

```
Input Video (B, 3, 32, 224, 224)
    ↓
ExplainableOpticalFlowResNet3D
├── RGB Stream + Attention
├── Optical Flow Stream
└── Feature Fusion
    ↓
┌─────────────────┬─────────────────┐
│ Diagnostic Head │  Subtype Head   │
│   (2 classes)   │   (4 classes)   │
└─────────────────┴─────────────────┘
```

### Previous (3-Head Model)

```
Input Video (B, 3, 32, 224, 224)
    ↓
ExplainableOpticalFlowResNet3D
├── RGB Stream + Attention
├── Optical Flow Stream
└── Feature Fusion
    ↓
┌─────────────┬─────────────┬──────────────────┐
│ Diagnostic  │  Subtype    │  Anatomical      │
│ (2 classes) │ (4 classes) │  (6 classes)     │ ← REMOVED
└─────────────┴─────────────┴──────────────────┘
```

## Training Example

```python
from multiclass_model import create_multiclass_model
from erdes_dataset import create_erdes_dataloaders
import torch.nn as nn

# Create model (no anatomical classes)
model = create_multiclass_model(
    num_diagnostic_classes=2,
    num_subtype_classes=4,
    pretrained=True
)

# Load data
train_loader, val_loader, test_loader = create_erdes_dataloaders(
    csv_path="../erdes/metadata.csv",
    data_root="../erdes"
)

# Training loop
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

for videos, labels, metadata in train_loader:
    outputs = model(videos)
    
    # Only two losses now
    loss_diagnostic = criterion(outputs['diagnostic'], labels['diagnostic'])
    loss_subtype = criterion(outputs['subtype'], labels['subtype'])
    
    loss = loss_diagnostic + loss_subtype  # No anatomical loss
    
    loss.backward()
    optimizer.step()
```

## VLM Prompts

### Before
```
The AI model predicts:
- Primary Diagnosis: rd (95%)
- Subtype: macula_detached (87%)
- Anatomical Location: TD (92%)  ← REMOVED

Explain the clinical reasoning...
```

### After
```
The AI model predicts:
- Primary Diagnosis: rd (95%)
- Subtype: macula_detached (87%)

Explain the clinical reasoning...
```

## Metadata Preservation

The `anatomical_subclass` is still available in metadata for reference:

```python
for videos, labels, metadata_list in dataloader:
    # labels only has 'diagnostic' and 'subtype'
    print(labels.keys())  # ['diagnostic', 'subtype']
    
    # But anatomical info is in metadata
    for metadata in metadata_list:
        print(metadata['anatomical_subclass'])  # 'TD', 'ND', 'N/A', etc.
```

## Benefits

1. **Simpler Model**: Fewer parameters, faster training
2. **Better Focus**: Model focuses on the two most important classifications
3. **Cleaner Data**: No need to handle anatomical labels during training
4. **Flexible**: Anatomical info still available in metadata if needed later

## Migration

If you have existing trained models with 3 heads, you'll need to:

1. **Retrain** with the new 2-head architecture
2. **Update** any inference code that expects anatomical predictions
3. **Modify** evaluation scripts to only check diagnostic and subtype

## Files Updated

- ✅ `multiclass_model.py` - Removed anatomical classifier
- ✅ `vlm_data_preparation.py` - Removed anatomical predictions
- ✅ `erdes_dataset.py` - Removed anatomical from labels
- ✅ `vlm_pipeline.py` - Removed anatomical parameters
- ✅ `REMOVED_ANATOMICAL.md` - This documentation

## Summary

The VLM system now trains on **2 classification tasks** instead of 3:
- ✅ Diagnostic class (non_rd vs rd)
- ✅ Subtype (normal, macula_intact, macula_detached, pvd)
- ❌ Anatomical subclass (removed from training)

All code is updated and ready to use! 🎉
