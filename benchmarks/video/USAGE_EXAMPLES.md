# Usage Examples

## Quick Test Script

### Default output location
```bash
./run_quick_test.sh
# Output: ./test_results/
```

### Custom output location
```bash
./run_quick_test.sh /path/to/my/results
# Output: /path/to/my/results/
```

### Examples
```bash
# Save to a specific project directory
./run_quick_test.sh ~/experiments/video_benchmark_test

# Save to a shared network drive
./run_quick_test.sh /mnt/shared/benchmarks/quick_test

# Save with descriptive name
./run_quick_test.sh ./results_quick_$(date +%Y%m%d)
```

## Full Benchmark Script

### Default output location (timestamped)
```bash
./run_full_benchmark.sh
# Output: ./results_YYYYMMDD_HHMMSS/
# Example: ./results_20260502_224500/
```

### Custom output location
```bash
./run_full_benchmark.sh /path/to/my/results
# Output: /path/to/my/results/
```

### Examples
```bash
# Save to a specific experiment directory
./run_full_benchmark.sh ~/experiments/video_benchmark_full

# Save to a shared network drive
./run_full_benchmark.sh /mnt/shared/benchmarks/full_run_v1

# Save with descriptive name
./run_full_benchmark.sh ./results_final_submission

# Save to Google Drive or similar
./run_full_benchmark.sh ~/GoogleDrive/research/benchmarks/video_classification
```

## Output Structure

Regardless of the output directory specified, the structure will be:

```
<output_directory>/
├── BENCHMARK_REPORT.md
├── comparison_table.csv
├── metrics_comparison.png
├── confusion_matrices.png
├── efficiency_analysis.png
├── radar_comparison.png
├── benchmark_results.json
├── models/
│   ├── resnet3d_best.pth
│   ├── i3d_best.pth
│   └── ...
├── logs/
│   ├── resnet3d_training.log
│   ├── resnet3d_history.json
│   └── ...
└── plots/
    ├── resnet3d_history.png
    ├── resnet3d_confusion_matrix.png
    └── ...
```

## Tips

### Organizing Multiple Runs
```bash
# Create a base directory for all experiments
mkdir -p ~/video_benchmarks

# Run different configurations
./run_full_benchmark.sh ~/video_benchmarks/run_baseline
./run_full_benchmark.sh ~/video_benchmarks/run_augmented
./run_full_benchmark.sh ~/video_benchmarks/run_longer_training
```

### Comparing Results from Different Runs
```bash
# After running multiple benchmarks, compare them
python compare_results.py --results_dir ~/video_benchmarks/run_baseline
python compare_results.py --results_dir ~/video_benchmarks/run_augmented

# Or manually compare the CSV files
diff ~/video_benchmarks/run_baseline/comparison_table.csv \
     ~/video_benchmarks/run_augmented/comparison_table.csv
```

### Archiving Results
```bash
# Compress results for sharing or archiving
tar -czf video_benchmark_results.tar.gz ./results_20260502_224500/

# Or use the custom directory name
./run_full_benchmark.sh ./results_paper_submission
tar -czf paper_submission_results.tar.gz ./results_paper_submission/
```
