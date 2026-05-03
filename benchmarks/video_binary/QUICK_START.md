# Quick Start Guide - Video Classification Benchmark

## 🚀 Quick Setup (3 steps)

### 1. Install Dependencies
```bash
cd benchmarks/video
pip install -r requirements.txt
```

### 2. Run Quick Test (5-10 minutes)
```bash
./run_quick_test.sh
```

### 3. Run Full Benchmark (4-8 hours)
```bash
./run_full_benchmark.sh
```

## 📊 What You Get

After running the benchmark, you'll have:

### Performance Metrics
- ✅ **Accuracy** - Overall classification accuracy
- ✅ **Precision** - Weighted precision score  
- ✅ **Recall** - Weighted recall score
- ✅ **F1 Score** - Weighted F1 score
- ✅ **AUC** - Area under ROC curve

### Visualizations
- 📈 **Metrics Comparison** - Bar charts comparing all metrics
- 🔥 **Confusion Matrices** - Per-model confusion matrices
- ⚡ **Efficiency Analysis** - Accuracy vs parameters and training time
- 🎯 **Radar Chart** - Multi-metric comparison

### Reports
- 📄 **Comparison Table** (CSV) - Easy to import into papers
- 📋 **Comprehensive Report** (Markdown) - Full analysis with recommendations

## 🎯 Models Included

| Model | Year | Type | Best For |
|-------|------|------|----------|
| **ResNet3D** | Baseline | 3D CNN | Standard baseline |
| **I3D** | 2017 | 3D CNN | General video classification |
| **SlowFast** | 2019 | Dual-pathway | Medical videos with motion |
| **X3D** | 2020 | Efficient 3D | Resource-constrained settings |
| **MViT** | 2021 | Transformer | High accuracy |
| **VideoMAE** | 2022 | Self-supervised | Limited data |
| **TimeSformer** | 2021 | Transformer | State-of-the-art |
| **C3D** | 2015 | Classic 3D | Medical imaging baseline |

## 🔧 Common Commands

### Train Specific Models Only
```bash
python train_benchmark.py --models resnet3d i3d mvit
```

### Use Different Dataset Split
```bash
python train_benchmark.py --split non_rd_vs_rd
```

### Adjust for GPU Memory
```bash
# For smaller GPU (reduce batch size and frames)
python train_benchmark.py --batch_size 4 --num_frames 16

# For larger GPU (increase batch size)
python train_benchmark.py --batch_size 16 --num_frames 32
```

### Quick Training (fewer epochs)
```bash
python train_benchmark.py --num_epochs 10
```

## 📁 Output Structure

```
results/
├── BENCHMARK_REPORT.md          ← Read this first!
├── comparison_table.csv          ← Import to Excel/Papers
├── metrics_comparison.png        ← Main comparison chart
├── efficiency_analysis.png       ← Model efficiency
├── radar_comparison.png          ← Multi-metric view
├── confusion_matrices.png        ← All confusion matrices
├── benchmark_results.json        ← Raw results
├── models/                       ← Trained weights
├── logs/                         ← Training logs
└── plots/                        ← Individual model plots
```

## 💡 Tips

### For Best Results
- Use **pretrained weights** (default: enabled)
- Train for **20 epochs** (default)
- Use **batch_size=8** for 224x224 images
- Enable **data augmentation** (default: enabled)

### For Faster Experimentation
- Reduce to **10 epochs**
- Use **batch_size=4**
- Test with **2-3 models** first
- Use **num_frames=16** instead of 32

### For Limited GPU Memory
```bash
python train_benchmark.py \
    --batch_size 2 \
    --num_frames 16 \
    --img_size 112
```

## 🐛 Troubleshooting

### Out of Memory Error
```bash
# Solution 1: Reduce batch size
python train_benchmark.py --batch_size 2

# Solution 2: Reduce frames
python train_benchmark.py --num_frames 16

# Solution 3: Reduce image size
python train_benchmark.py --img_size 112
```

### Slow Data Loading
```bash
# Increase workers
python train_benchmark.py --num_workers 8
```

### CUDA Out of Memory During Validation
```bash
# Reduce batch size affects both training and validation
python train_benchmark.py --batch_size 4
```

## 📊 Expected Performance

Based on similar medical video datasets, you can expect:

- **Accuracy**: 70-95% (depends on data quality)
- **F1 Score**: 0.70-0.95
- **Training Time**: 10-60 min per model (20 epochs)

Top performers are typically:
1. **MViT** or **TimeSformer** (highest accuracy)
2. **SlowFast** (good for medical videos)
3. **I3D** (strong baseline)

## 🎓 Understanding Results

### Accuracy vs F1 Score
- **Accuracy**: Overall correctness
- **F1 Score**: Better for imbalanced datasets (medical data often is)
- **Recommendation**: Prioritize F1 for medical applications

### Model Size vs Performance
- Larger models (MViT, TimeSformer): Higher accuracy, slower
- Smaller models (X3D, ResNet3D): Faster, good baseline
- **Recommendation**: Start with medium models (I3D, SlowFast)

### When to Use Which Model

**For Production/Deployment:**
- Use **X3D** or **ResNet3D** (efficient, fast inference)

**For Research/Best Performance:**
- Use **MViT** or **TimeSformer** (state-of-the-art)

**For Limited Training Data:**
- Use **VideoMAE** (self-supervised pretraining)

**For Medical Videos with Motion:**
- Use **SlowFast** (dual-pathway captures motion well)

## 📞 Need Help?

Check these files:
1. `README.md` - Full documentation
2. `BENCHMARK_REPORT.md` - Generated after training
3. Training logs in `results/logs/`

## 🎉 Next Steps

After running the benchmark:

1. **Review** `BENCHMARK_REPORT.md`
2. **Compare** models in `comparison_table.csv`
3. **Visualize** with the generated PNG files
4. **Select** best model for your use case
5. **Deploy** using weights in `models/` directory

Good luck! 🚀
