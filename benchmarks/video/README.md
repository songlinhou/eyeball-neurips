# Video Classification Benchmark

This directory contains scripts for benchmarking state-of-the-art video classification methods on medical video data (ERDES dataset).

## Overview

This benchmark compares the following video classification models:

### State-of-the-Art Models

1. **I3D (Inflated 3D ConvNet)** - CVPR 2017
   - Inflates 2D filters to 3D for video understanding
   - Widely used baseline for video classification

2. **SlowFast Networks** - ICCV 2019
   - Dual-pathway architecture: slow pathway for spatial semantics, fast for motion
   - Popular for medical video analysis due to multi-scale temporal modeling

3. **X3D** - CVPR 2020
   - Efficient video network with progressive expansion
   - Good for resource-constrained medical applications

4. **TimeSformer** - ICML 2021
   - Transformer-based with divided space-time attention
   - State-of-the-art for many video tasks

5. **VideoMAE** - NeurIPS 2022
   - Self-supervised learning approach with masked autoencoders
   - Excellent for limited medical data scenarios

6. **MViT (Multiscale Vision Transformers)** - ICCV 2021
   - Hierarchical vision transformer for video recognition
   - Efficient and accurate for medical video classification

7. **C3D** - ICCV 2015
   - Classic 3D CNN baseline
   - Widely used in medical imaging

8. **ResNet3D** - Baseline
   - Standard 3D ResNet for video classification
   - Strong baseline performance

## Dataset

The benchmark uses the ERDES (Eye Retinal Detachment Ultrasound) dataset with the same configuration as the main experiments:

- **Task**: Macula detached vs intact classification
- **Data**: 3D ocular ultrasound videos
- **Splits**: Train/Val/Test from `erdes/splits/macula_detached_vs_intact/`
- **Settings**: Same as `/home/ray/research/eyeball-llm/eyeball-neurips/model/run_experiments.py`

## Installation

```bash
cd benchmarks/video
pip install -r requirements.txt
```

## Usage

### Run Full Benchmark

Train and evaluate all models:

```bash
python train_benchmark.py \
    --data_dir ../../erdes \
    --split macula_detached_vs_intact \
    --save_dir ./results \
    --num_frames 32 \
    --img_size 224 \
    --batch_size 8 \
    --num_epochs 20 \
    --learning_rate 1e-4 \
    --num_workers 4
```

### Run Specific Models

Train only selected models:

```bash
python train_benchmark.py \
    --models resnet3d i3d mvit \
    --save_dir ./results \
    --num_epochs 20
```

### Compare Results

Generate comparison visualizations and report:

```bash
python compare_results.py \
    --results_dir ./results \
    --results_file benchmark_results.json
```

## Configuration

### Command Line Arguments

**train_benchmark.py:**
- `--data_dir`: Path to data directory (default: `../../erdes`)
- `--split`: Dataset split to use (default: `macula_detached_vs_intact`)
- `--save_dir`: Directory to save results (default: `./results`)
- `--models`: Models to test (default: all models)
- `--num_frames`: Number of frames per video (default: 32)
- `--img_size`: Image size (default: 224)
- `--batch_size`: Batch size (default: 8)
- `--num_epochs`: Number of training epochs (default: 20)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--weight_decay`: Weight decay (default: 1e-4)
- `--dropout`: Dropout rate (default: 0.5)
- `--num_workers`: Number of data loading workers (default: 4)
- `--pretrained`: Use pretrained weights (default: True)

**compare_results.py:**
- `--results_dir`: Directory containing benchmark results (default: `./results`)
- `--results_file`: Name of results JSON file (default: `benchmark_results.json`)

## Output

The benchmark generates the following outputs in the `save_dir`:

### Directory Structure
```
results/
├── models/                          # Trained model weights
│   ├── resnet3d_best.pth
│   ├── i3d_best.pth
│   └── ...
├── logs/                            # Training logs and metrics
│   ├── resnet3d_training.log
│   ├── resnet3d_history.json
│   └── ...
├── plots/                           # Individual model plots
│   ├── resnet3d_history.png
│   ├── resnet3d_confusion_matrix.png
│   └── ...
├── benchmark_results.json           # Complete results JSON
├── comparison_table.csv             # Results comparison table
├── metrics_comparison.png           # All metrics comparison
├── confusion_matrices.png           # All confusion matrices
├── efficiency_analysis.png          # Accuracy vs params/time
├── radar_comparison.png             # Radar chart comparison
└── BENCHMARK_REPORT.md             # Comprehensive report
```

### Metrics Reported

For each model, the following metrics are computed:
- **Accuracy**: Overall classification accuracy
- **Precision**: Weighted precision score
- **Recall**: Weighted recall score
- **F1 Score**: Weighted F1 score
- **AUC**: Area under ROC curve
- **Parameters**: Number of model parameters
- **Training Time**: Total training time in minutes
- **Confusion Matrix**: Per-class performance

## Example Results

After running the benchmark, you can view:

1. **Comparison Table** (`comparison_table.csv`):
   - Side-by-side comparison of all metrics
   - Easy to identify best performing models

2. **Visualizations**:
   - Bar charts comparing accuracy, precision, recall, F1, AUC
   - Scatter plots for efficiency analysis
   - Radar charts for multi-metric comparison
   - Confusion matrices for all models

3. **Comprehensive Report** (`BENCHMARK_REPORT.md`):
   - Overview of all models
   - Performance summary
   - Best performing models
   - Key findings and recommendations

## Training Strategy

All models use a two-phase training strategy (same as main experiments):

### Phase 1: Classifier Head Training (5 epochs)
- Freeze backbone weights
- Train only the classification head
- Higher learning rate (10x base LR)
- Cosine annealing with warm restarts

### Phase 2: Full Fine-tuning (15 epochs)
- Unfreeze all weights
- Fine-tune entire model
- Base learning rate
- ReduceLROnPlateau scheduler
- Early stopping (patience=7)

## Medical Video Classification Considerations

These models are specifically chosen for medical video analysis because:

1. **Temporal Modeling**: All models capture temporal dynamics crucial for medical videos
2. **Pretrained Weights**: Transfer learning from Kinetics-400 helps with limited medical data
3. **Efficiency**: Range of model sizes for different computational budgets
4. **Interpretability**: Some models (TimeSformer, MViT) provide attention visualizations
5. **Proven Performance**: All models have shown success in medical imaging tasks

## Tips for Best Results

1. **Batch Size**: Adjust based on GPU memory (reduce if OOM errors occur)
2. **Number of Frames**: More frames capture more temporal context but require more memory
3. **Image Size**: 224x224 is standard, but can be reduced for efficiency
4. **Epochs**: 20 epochs is usually sufficient with early stopping
5. **Data Augmentation**: Already enabled in the dataset loader
6. **Class Imbalance**: Handled via weighted loss function

## Troubleshooting

### Out of Memory (OOM)
- Reduce `--batch_size` (try 4 or 2)
- Reduce `--num_frames` (try 16)
- Reduce `--img_size` (try 112)
- Use gradient accumulation

### Slow Training
- Increase `--num_workers` for faster data loading
- Use mixed precision training (add to code if needed)
- Reduce model complexity (use smaller variants)

### Poor Performance
- Increase `--num_epochs`
- Adjust `--learning_rate` (try 5e-5 or 2e-4)
- Check data quality and class balance
- Try different models

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@article{ozkuterdes,
  title={ERDES: A Benchmark Video Dataset for Retinal Detachment and Macular Status Classification in Ocular Ultrasound},
  author={Ozkut, Yasemin and Navard, Pouyan and Adhikari, Srikar and Situ-LaCasse, Elaine and Acu{\~n}a, Josie and Yarnish, Adrienne A and Yilmaz, Alper},
  journal={arXiv preprint arXiv:2508.04735},
  year={2025}
}
```

## References

- **I3D**: Carreira & Zisserman. "Quo Vadis, Action Recognition?" CVPR 2017
- **SlowFast**: Feichtenhofer et al. "SlowFast Networks for Video Recognition" ICCV 2019
- **X3D**: Feichtenhofer. "X3D: Expanding Architectures for Efficient Video Recognition" CVPR 2020
- **TimeSformer**: Bertasius et al. "Is Space-Time Attention All You Need for Video Understanding?" ICML 2021
- **VideoMAE**: Tong et al. "VideoMAE: Masked Autoencoders are Data-Efficient Learners" NeurIPS 2022
- **MViT**: Fan et al. "Multiscale Vision Transformers" ICCV 2021
- **C3D**: Tran et al. "Learning Spatiotemporal Features with 3D CNNs" ICCV 2015
