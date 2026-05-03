# Video Classification Benchmark - File Index

## 📚 Documentation Files

### Start Here
1. **`QUICK_START.md`** ⭐ - Get started in 3 steps (5 min read)
2. **`SUMMARY.md`** - Complete overview of what's included
3. **`README.md`** - Full documentation and reference

### This File
- **`INDEX.md`** - You are here! Navigation guide

## 🔧 Core Implementation Files

### Models
- **`video_models.py`** (560 lines)
  - 8 video classification models
  - Factory function `get_model()`
  - All models support freeze/unfreeze for transfer learning

### Training
- **`train_benchmark.py`** (580 lines)
  - Complete training pipeline
  - Two-phase training strategy
  - Automatic evaluation and logging
  - Command-line interface

### Analysis
- **`compare_results.py`** (520 lines)
  - Generate comparison visualizations
  - Create comprehensive reports
  - Export results to CSV/JSON/Markdown

### Package
- **`__init__.py`** - Python package initialization
- **`requirements.txt`** - Python dependencies

## 🚀 Executable Scripts

### Quick Test (Recommended First)
- **`run_quick_test.sh`** 
  - Tests 2 models (ResNet3D, I3D)
  - 5 epochs, 16 frames
  - Takes 5-10 minutes
  - Verifies setup works

### Full Benchmark
- **`run_full_benchmark.sh`**
  - All 8 models
  - 20 epochs, 32 frames
  - Takes 4-8 hours
  - Production results

## 📖 How to Use This Benchmark

### First Time Users
```
1. Read: QUICK_START.md (5 min)
2. Install: pip install -r requirements.txt
3. Test: ./run_quick_test.sh (10 min)
4. Review: Check test_results/ folder
```

### Running Full Benchmark
```
1. Read: SUMMARY.md (understand what you're running)
2. Run: ./run_full_benchmark.sh (4-8 hours)
3. Analyze: Check results_TIMESTAMP/ folder
4. Report: Read BENCHMARK_REPORT.md
```

### Custom Experiments
```
1. Read: README.md (full documentation)
2. Modify: train_benchmark.py arguments
3. Run: python train_benchmark.py --help
4. Compare: python compare_results.py
```

## 🎯 File Purposes

| File | Purpose | When to Use |
|------|---------|-------------|
| `QUICK_START.md` | Get started quickly | First time setup |
| `SUMMARY.md` | Understand what's included | Before running benchmark |
| `README.md` | Full reference | Detailed configuration |
| `video_models.py` | Model implementations | Adding new models |
| `train_benchmark.py` | Training pipeline | Running experiments |
| `compare_results.py` | Analysis tools | After training |
| `run_quick_test.sh` | Quick verification | Testing setup |
| `run_full_benchmark.sh` | Full benchmark | Production run |
| `requirements.txt` | Dependencies | Installation |

## 📊 Output Files (Generated)

After running the benchmark, you'll get:

### In `results/` or `test_results/`

#### Reports
- `BENCHMARK_REPORT.md` - Comprehensive analysis
- `comparison_table.csv` - Results table
- `benchmark_results.json` - Raw results

#### Visualizations
- `metrics_comparison.png` - All metrics bar charts
- `confusion_matrices.png` - All confusion matrices
- `efficiency_analysis.png` - Accuracy vs params/time
- `radar_comparison.png` - Multi-metric radar chart

#### Model Outputs
- `models/*.pth` - Trained model weights
- `logs/*.log` - Training logs
- `logs/*.json` - Training history
- `plots/*.png` - Per-model plots

## 🔍 Quick Reference

### Models Available
```
resnet3d    - Standard 3D ResNet baseline
i3d         - Inflated 3D ConvNet (CVPR 2017)
slowfast    - Dual-pathway network (ICCV 2019)
x3d         - Efficient video network (CVPR 2020)
mvit        - Multiscale Vision Transformer (ICCV 2021)
videomae    - Masked Autoencoder (NeurIPS 2022)
timesformer - Space-time attention (ICML 2021)
c3d         - Classic 3D CNN (ICCV 2015)
```

### Common Commands
```bash
# Quick test
./run_quick_test.sh

# Full benchmark
./run_full_benchmark.sh

# Specific models
python train_benchmark.py --models resnet3d i3d

# Custom config
python train_benchmark.py --batch_size 4 --num_epochs 10

# Compare results
python compare_results.py --results_dir ./results

# Help
python train_benchmark.py --help
```

### Metrics Computed
```
✓ Accuracy    - Overall classification accuracy
✓ Precision   - Weighted precision score
✓ Recall      - Weighted recall score
✓ F1 Score    - Weighted F1 score
✓ AUC         - Area under ROC curve
✓ Confusion   - Confusion matrix
✓ Params      - Model parameter count
✓ Time        - Training time
```

## 🎓 Learning Path

### Beginner
1. Read `QUICK_START.md`
2. Run `./run_quick_test.sh`
3. Explore `test_results/`
4. Read generated `BENCHMARK_REPORT.md`

### Intermediate
1. Read `SUMMARY.md`
2. Understand model architectures in `video_models.py`
3. Run `./run_full_benchmark.sh`
4. Analyze all visualizations

### Advanced
1. Read `README.md` fully
2. Study `train_benchmark.py` implementation
3. Modify models or add new ones
4. Customize training pipeline
5. Create custom analysis scripts

## 🔧 Troubleshooting Guide

### Problem: Out of Memory
**Solution**: Reduce batch size or frames
```bash
python train_benchmark.py --batch_size 2 --num_frames 16
```

### Problem: Slow Training
**Solution**: Increase workers or reduce epochs
```bash
python train_benchmark.py --num_workers 8 --num_epochs 10
```

### Problem: Poor Results
**Solution**: Check data, increase epochs, adjust LR
```bash
python train_benchmark.py --num_epochs 30 --learning_rate 5e-5
```

### Problem: Missing Dependencies
**Solution**: Reinstall requirements
```bash
pip install -r requirements.txt --upgrade
```

## 📞 Support Resources

### Documentation
- `README.md` - Full documentation
- `QUICK_START.md` - Quick start guide
- `SUMMARY.md` - Overview

### Code
- `video_models.py` - Model implementations
- `train_benchmark.py` - Training pipeline
- `compare_results.py` - Analysis tools

### Examples
- `run_quick_test.sh` - Quick test example
- `run_full_benchmark.sh` - Full benchmark example

## ✅ Checklist

Before running benchmark:
- [ ] Read `QUICK_START.md`
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Check GPU availability: `nvidia-smi`
- [ ] Verify data path: `../../erdes/splits/macula_detached_vs_intact/`

After running benchmark:
- [ ] Check `BENCHMARK_REPORT.md`
- [ ] Review `comparison_table.csv`
- [ ] Examine visualizations (PNG files)
- [ ] Verify model weights saved in `models/`

## 🎯 Next Steps

1. **Start**: Run `./run_quick_test.sh`
2. **Learn**: Read generated reports
3. **Experiment**: Try different configurations
4. **Deploy**: Use best model for your application
5. **Extend**: Add new models or datasets

## 📝 Notes

- All scripts use the same dataset and settings as `model/run_experiments.py`
- Results are reproducible (can set random seeds)
- Models are pretrained on Kinetics-400 by default
- Training uses two-phase strategy (freeze → unfreeze)
- Early stopping prevents overfitting
- Class weighting handles imbalanced data

---

**Ready to start?** → Read `QUICK_START.md` and run `./run_quick_test.sh`! 🚀
