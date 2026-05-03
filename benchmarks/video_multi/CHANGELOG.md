# Changelog - Multi-Class Video Benchmark

## Version 2.1 - Added Proposed Method (May 3, 2026)

### Added
- **Proposed Method**: `explainable` model now included in default benchmark
  - Multi-class ExplainableResNet3D from `/model/multiclass_model.py`
  - Features: Temporal/spatial attention + optical flow + frame importance
  - Expected to outperform baselines (87-93% diagnostic, 75-85% subtype)

### Changed
- **Total Models**: 8 → 9 models (8 baselines + 1 proposed)
- **Default Benchmark**: Now includes proposed method
- **Documentation**: Updated to highlight proposed vs baseline models

### Model Details
The proposed `explainable` model includes:
- ✅ Temporal attention (from exp07_improved_lower_dropout)
- ✅ CBAM3D (channel + spatial attention)
- ✅ Optical flow extraction
- ✅ Frame importance module
- ✅ Spatial explainability module
- ✅ Dual classification heads (diagnostic + subtype)

---

## Version 2.0 - Full Model Suite (May 3, 2026)

### Added
- **5 New Models** to match the binary benchmark:
  - `slowfast` - Dual-pathway architecture (ICCV 2019)
  - `x3d` - Efficient video network (CVPR 2020)
  - `videomae` - Masked autoencoder (NeurIPS 2022)
  - `timesformer` - Space-time attention (ICML 2021)
  - `c3d` - Classic 3D CNN (ICCV 2015)

### Changed
- **Total Models**: Increased from 4 to 8 models
- **Default Benchmark**: Now runs all 8 models by default
- **Model Factory**: Updated `get_multiclass_model()` to support all 8 models
- **Documentation**: Updated all docs to reflect 8-model suite

### Model Classes Added
1. `MultiClassSlowFast` - Dual-pathway with multi-class heads
2. `MultiClassX3D` - Efficient network with multi-class heads
3. `MultiClassVideoMAE` - Masked autoencoder with multi-class heads
4. `MultiClassTimeSformer` - Space-time attention with multi-class heads
5. `MultiClassC3D` - Classic 3D CNN with multi-class heads

### Files Modified
- `multiclass_models.py` - Added 5 new model classes (~350 lines added)
- `train_benchmark.py` - Updated default models list and factory call
- `run_full_benchmark.sh` - Updated to benchmark all 8 models
- `README.md` - Updated model list and examples
- `INDEX.md` - Updated model descriptions and performance table
- `QUICK_REFERENCE.md` - Updated model table and examples

### Compatibility
- ✅ Backward compatible - existing code still works
- ✅ All models use same dual-head architecture
- ✅ Same training pipeline for all models
- ✅ Consistent evaluation metrics

### Performance
Expected training time for full benchmark (8 models, 20 epochs):
- **Total**: ~6-8 hours on single GPU
- **Per model**: 30-75 minutes depending on architecture

### Usage
```bash
# Run all 8 models
./run_full_benchmark.sh

# Run specific models
python train_benchmark.py --models resnet3d i3d slowfast x3d

# Quick test with new models
python train_benchmark.py --models slowfast --num_epochs 5
```

---

## Version 1.0 - Initial Release (May 3, 2026)

### Initial Features
- Multi-class hierarchical classification
- 4 initial models: ResNet3D, I3D, MViT, Explainable
- Reproducible train/test splits (300/100)
- Comprehensive evaluation metrics
- Automated comparison and reporting
