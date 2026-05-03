# Multi-Class Video Benchmark - Quick Reference

## 🚀 Quick Start (3 Commands)

```bash
# 1. Prepare splits
python prepare_splits.py

# 2. Run benchmark
./run_full_benchmark.sh

# 3. Analyze results
python compare_results.py
```

## 📊 Classification Tasks

| Task | Classes | Labels |
|------|---------|--------|
| **Diagnostic** | 2 | non_rd (0), rd (1) |
| **Subtype** | 4 | macula_detached (0), macula_intact (1), normal (2), pvd (3) |

## 🎯 Dataset Split

- **Training**: 300 samples (stratified by subtype)
- **Testing**: 100 samples (stratified by subtype)
- **Random Seed**: 42 (reproducible)

## 🤖 Supported Models (9 Total)

| Model | Description | Year | Parameters |
|-------|-------------|------|------------|
| `resnet3d` | Baseline 3D ResNet | 2018 | ~33M |
| `i3d` | Inflated 3D ConvNet | 2017 | ~33M |
| `slowfast` | Dual-pathway | 2019 | ~66M |
| `x3d` | Efficient network | 2020 | ~35M |
| `mvit` | Vision Transformer | 2021 | ~35M |
| `videomae` | Masked autoencoder | 2022 | ~35M |
| `timesformer` | Space-time attention | 2021 | ~120M |
| `c3d` | Classic 3D CNN | 2015 | ~78M |
| `explainable` | **PROPOSED** - Attention + Flow | 2026 | ~37M |

## ⚙️ Key Parameters

```bash
--num_frames 32        # Frames per video
--img_size 224         # Image resolution
--batch_size 8         # Batch size
--num_epochs 20        # Training epochs
--learning_rate 1e-4   # Learning rate
--dropout 0.5          # Dropout rate
```

## 📁 Output Structure

```
results/
├── models/              # Trained weights (.pth)
├── logs/                # Training logs (.log, .json)
├── plots/               # Visualizations (.png)
├── benchmark_results.json
├── comparison_table.csv
└── BENCHMARK_REPORT.md
```

## 📈 Evaluation Metrics

**Per Task (Diagnostic & Subtype):**
- Accuracy (%)
- Precision (weighted)
- Recall (weighted)
- F1 Score (weighted)
- Confusion Matrix

## 🔧 Common Commands

### Train Single Model
```bash
python train_benchmark.py --models resnet3d --num_epochs 20
```

### Train Multiple Models
```bash
python train_benchmark.py --models resnet3d i3d slowfast x3d
```

### Quick Test (Fast)
```bash
./run_quick_test.sh  # 1 model, 5 epochs, 16 frames
```

### Custom Configuration
```bash
python train_benchmark.py \
    --models slowfast videomae \
    --num_epochs 30 \
    --batch_size 4 \
    --num_frames 16 \
    --learning_rate 5e-5
```

### Analyze Results
```bash
python compare_results.py --results_dir ./results
```

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Out of memory | `--batch_size 4 --num_frames 16` |
| Slow training | `--num_workers 8` |
| Poor accuracy | `--num_epochs 30 --learning_rate 5e-5` |

## 📋 File Checklist

- [x] `prepare_splits.py` - Data preparation
- [x] `multiclass_dataset.py` - Dataset loader
- [x] `multiclass_models.py` - Model definitions
- [x] `train_benchmark.py` - Training script
- [x] `compare_results.py` - Results analysis
- [x] `run_quick_test.sh` - Quick test
- [x] `run_full_benchmark.sh` - Full benchmark
- [x] `README.md` - Full documentation
- [x] `INDEX.md` - Complete index
- [x] `SUMMARY.md` - Overview
- [x] `requirements.txt` - Dependencies

## 🔗 Related Files

- **Model**: `/home/ray/research/eyeball-llm/eyeball-neurips/model/multiclass_model.py`
- **Data**: `/home/ray/research/eyeball-llm/eyeball-neurips/erdes/metadata.csv`
- **Binary Benchmark**: `/home/ray/research/eyeball-llm/eyeball-neurips/benchmarks/video_binary/`

## ⏱️ Expected Runtime

| Configuration | Time |
|---------------|------|
| Quick test (1 model, 5 epochs) | ~10-15 min |
| Single model (20 epochs) | ~30-60 min |
| Full benchmark (4 models) | ~2-4 hours |

*Times vary based on GPU and configuration*

## 💡 Tips

1. **Start with quick test** to verify setup
2. **Use smaller batch_size** if GPU memory limited
3. **Monitor training logs** in real-time
4. **Compare with binary benchmark** for insights
5. **Save best models** for downstream tasks

## 📞 Help

- See `README.md` for detailed usage
- See `INDEX.md` for complete reference
- See `SUMMARY.md` for overview
- Check logs in `results/logs/` for errors

---

**Ready to start?** Run: `./run_quick_test.sh`
