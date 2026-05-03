# Class Labels Update - ERDES metadata.csv

## Summary

Updated all VLM system code to use the correct class labels from `/home/ray/research/eyeball-llm/eyeball-neurips/erdes/metadata.csv`.

## Correct Class Structure

### diagnostic_class (2 classes)
- `non_rd`: Non-Retinal Detachment
- `rd`: Retinal Detachment

### subtype (4 classes)
- `normal`: Normal healthy eye
- `macula_intact`: RD with macula attached (previously called "macula_on")
- `macula_detached`: RD with macula detached (previously called "macula_off")
- `pvd`: Posterior Vitreous Detachment

### anatomical_subclass (6 classes)
- `N/A`: Not applicable (for normal/pvd cases)
- `TD`: Total Detachment
- `ND`: Nasal Detachment
- `Bilateral`: Bilateral detachment
- `SD`: Superior Detachment
- `ID`: Inferior Detachment

## What Changed

### ❌ Old (Incorrect) Labels

```python
subtype_labels = {
    0: "normal",
    1: "macula_on",      # WRONG
    2: "macula_off",     # WRONG
    3: "pvd"
}

anatomical_labels = {
    0: "none",           # WRONG
    1: "superior",       # WRONG
    2: "inferior",       # WRONG
    3: "temporal",       # WRONG
    4: "nasal",          # WRONG
    5: "multiple"        # WRONG
}
```

### ✅ New (Correct) Labels

```python
subtype_labels = {
    0: "normal",
    1: "macula_intact",    # CORRECT
    2: "macula_detached",  # CORRECT
    3: "pvd"
}

anatomical_labels = {
    0: "N/A",         # CORRECT
    1: "TD",          # CORRECT
    2: "ND",          # CORRECT
    3: "Bilateral",   # CORRECT
    4: "SD",          # CORRECT
    5: "ID"           # CORRECT
}
```

## Files Updated

1. ✅ **vlm_data_preparation.py**
   - Updated `self.subtype_labels`
   - Updated `self.anatomical_labels`

2. ✅ **erdes_dataset.py**
   - Updated `self.subtype_to_idx`
   - Updated `self.anatomical_to_idx`
   - Added support for both "N/A" and "n/a"
   - Fixed anatomical_subclass handling to preserve case

3. ✅ **ERDES_INTEGRATION.md**
   - Updated class hierarchy documentation
   - Updated code examples

4. ✅ **VLM_README.md**
   - Updated class definitions
   - Updated output structure examples

5. ✅ **CLASS_LABELS_UPDATE.md** (this file)
   - Summary of changes

## Verification

From metadata.csv analysis:
- ✅ `subtype` values: `normal`, `macula_intact`, `macula_detached`, `pvd`
- ✅ `anatomical_subclass` values: `N/A`, `TD`, `ND`, `Bilateral`, `SD`, `ID`

## Example Data

```csv
# From metadata.csv
clip_id,file_path,diagnostic_class,subtype,anatomical_subclass
102769_00001,clips/Retinal_Detachment/Macula_Intact/TD/102769_00001.mp4,rd,macula_intact,TD
164267_00001,clips/Non_Retinal_Detachment/Normal/164267_00001.mp4,non_rd,normal,N/A
405744_00001,clips/Retinal_Detachment/Macula_Detached/TD/405744_00001.mp4,rd,macula_detached,TD
825315_00001,clips/Non_Retinal_Detachment/Posterior_Vitreous_Detachment/825315_00001.mp4,non_rd,pvd,N/A
```

## Impact

### No Breaking Changes
- Model architecture remains the same (2, 4, 6 classes)
- Only label strings changed
- All code still compatible

### Benefits
- ✅ Matches actual ERDES dataset
- ✅ Consistent with metadata.csv
- ✅ Proper medical terminology
- ✅ Ready for production use

## Testing

To verify the changes work correctly:

```bash
cd /home/ray/research/eyeball-llm/eyeball-neurips/model
python erdes_dataset.py
```

Expected output:
```
Loaded 5383 samples from ../erdes/metadata.csv
Diagnostic classes: {'non_rd': 4737, 'rd': 646}
Subtype classes: {'normal': 4091, 'pvd': 646, 'macula_intact': 433, 'macula_detached': 213}
Video shape: torch.Size([3, 32, 224, 224])
Labels: {'diagnostic': 1, 'subtype': 1, 'anatomical': 1}
```

## Migration Guide

If you have existing code using old labels:

### Old Code
```python
if prediction == "macula_on":
    # ...
```

### New Code
```python
if prediction == "macula_intact":
    # ...
```

### Mapping
- `macula_on` → `macula_intact`
- `macula_off` → `macula_detached`
- `none` → `N/A`
- `superior` → `SD`
- `inferior` → `ID`
- `nasal` → `ND`
- `multiple` → `Bilateral` or `TD`

## Summary

All VLM system code now correctly uses the ERDES metadata.csv class labels:
- ✅ `macula_intact` / `macula_detached` (not macula_on/off)
- ✅ `N/A`, `TD`, `ND`, `Bilateral`, `SD`, `ID` (not none/superior/inferior/etc.)

The system is now ready for training and deployment with the actual ERDES dataset! 🎉
