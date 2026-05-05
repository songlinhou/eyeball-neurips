# Figure 2: Attention Visualization Examples

This directory contains scripts to generate Figure 2 for the paper, which demonstrates the interpretability of the model through attention visualizations.

## Overview

Figure 2 shows:
- **Top-5 most important frames** with spatial attention heatmaps overlaid
- **Frame importance scores** across all frames (bar chart)
- **Predictions vs. ground truth** for diagnostic and subtype classification
- **Multiple diverse examples** covering different classes and scenarios

## Files

- `generate_figure2_attention.py` - Main Python script for generating visualizations
- `run_figure2.sh` - Shell script for easy execution
- `README_FIGURE2.md` - This file

## Requirements

```bash
pip install numpy pandas matplotlib seaborn opencv-python torch
```

## Quick Start

### Using the Shell Script (Recommended)

```bash
cd /home/ray/research/eyeball-llm/eyeball-neurips/visualization
bash run_figure2.sh
```

### Using Python Directly

```bash
python generate_figure2_attention.py \
    --model_checkpoint ../model/checkpoints/multiclass/best_model_weights.pth \
    --data_csv ../benchmarks/input/balanced_split_desc.csv \
    --video_base_dir ../benchmarks/input \
    --output_dir ./figures \
    --num_examples 6 \
    --device cuda
```

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model_checkpoint` | str | Required | Path to trained model checkpoint (.pth) |
| `--data_csv` | str | Required | Path to dataset CSV file |
| `--video_base_dir` | str | None | Base directory for video files (if paths are relative) |
| `--output_dir` | str | `./figures` | Output directory for generated figures |
| `--num_examples` | int | 6 | Number of examples to visualize |
| `--device` | str | `cuda` | Device to use (cuda/cpu) |
| `--num_frames` | int | 32 | Number of frames to sample from each video |
| `--img_size` | int | 224 | Image size for preprocessing |

## Output Files

The script generates two types of outputs:

### 1. Individual Visualizations
- **Filename**: `attention_<video_name>.png`
- **Layout**: 
  - Top row: 5 key frames with spatial attention overlays
  - Middle row: Frame importance bar chart
  - Bottom row: Prediction and ground truth information
- **Size**: ~20x10 inches at 300 DPI

### 2. Combined Figure 2
- **Filename**: `Figure2_Attention_Visualization.png`
- **Layout**: Compact grid showing all examples
- **Columns**:
  1. Top 1 frame (highest importance)
  2. Top 2 frame
  3. Top 3 frame
  4. Top 4 frame
  5. Top 5 frame
  6. Frame importance chart
  7. Info panel (predictions, ground truth, status)
- **Size**: ~24x(4*num_examples) inches at 300 DPI

## Example Selection Strategy

The script automatically selects diverse examples covering:

1. **RD + Macula Detached** - Most urgent surgical case
2. **RD + Macula Intact** - Urgent but better prognosis
3. **Non-RD + Normal** - Healthy eye
4. **Non-RD + PVD** - Common benign condition
5-6. **Additional diverse samples** - Random selection from remaining data

This ensures the figure shows the model's performance across all major clinical scenarios.

## Customization

### Change Number of Examples

```bash
python generate_figure2_attention.py \
    --model_checkpoint <path> \
    --data_csv <path> \
    --num_examples 4  # Show only 4 examples
```

### Select Specific Videos

Modify the `select_diverse_examples()` function in the script to manually specify video indices or names.

### Adjust Heatmap Appearance

In the `overlay_heatmap()` function:
- `alpha=0.5` - Controls transparency (0=invisible, 1=opaque)
- `colormap=cv2.COLORMAP_JET` - Color scheme (JET, HOT, VIRIDIS, etc.)

### Change Figure Size

In `create_combined_figure()`:
```python
fig = plt.figure(figsize=(24, 4 * num_examples))  # Adjust width and height
```

## Troubleshooting

### "Model checkpoint not found"
- Ensure you've trained the model first using `run_training.sh`
- Check the path to `best_model_weights.pth`

### "Video file not found"
- Verify `--video_base_dir` points to the correct location
- Check that video paths in CSV are correct

### "CUDA out of memory"
- Use `--device cpu` to run on CPU
- Reduce `--num_examples` to process fewer videos

### "Import error: No module named 'multiclass_model'"
- Ensure you're running from the `visualization/` directory
- The script adds `../model` to the Python path automatically

## Example Output

```
================================================================================
Generating Figure 2: Attention Visualization Examples
================================================================================
Model: ../model/checkpoints/multiclass/best_model_weights.pth
Data: ../benchmarks/input/balanced_split_desc.csv
Output: ./figures
Device: cuda
Examples: 6
================================================================================

Loading model...
Model loaded successfully

Loading dataset...
Selected 6 diverse examples

Processing: 164267_02030
  ✓ Processed successfully

Processing: 164267_01439
  ✓ Processed successfully

...

Saved combined Figure 2 to: ./figures/Figure2_Attention_Visualization.png

================================================================================
Figure generation complete!
Individual visualizations: ./figures/attention_*.png
Combined Figure 2: ./figures/Figure2_Attention_Visualization.png
================================================================================
```

## Integration with Paper

### LaTeX Code

```latex
\begin{figure*}[t]
    \centering
    \includegraphics[width=\textwidth]{figures/Figure2_Attention_Visualization.png}
    \caption{
        \textbf{Attention Visualization Examples.} 
        Our model provides interpretable predictions through intrinsic attention mechanisms.
        For each example, we show the top-5 most important frames (columns 1-5) with spatial 
        attention heatmaps overlaid, frame importance scores across all frames (column 6), 
        and prediction vs. ground truth information (column 7). 
        The model correctly identifies diagnostically relevant regions (e.g., detached retina, 
        macula status) and assigns higher importance to frames containing key pathological features.
        Examples include: (Row 1) RD with macula detached, (Row 2) RD with macula intact, 
        (Row 3) Normal eye, (Row 4) PVD, (Rows 5-6) Additional diverse cases.
    }
    \label{fig:attention_visualization}
\end{figure*}
```

## Tips for Best Results

1. **Use high-quality model checkpoint**: Ensure the model is fully trained
2. **Select diverse examples**: Include both correct and challenging cases
3. **Verify video quality**: Use clear, well-captured ultrasound videos
4. **Check color rendering**: Ensure heatmaps are visible and informative
5. **Adjust DPI for publication**: Use 300 DPI for print, 150 DPI for digital

## Future Enhancements

Potential improvements to the visualization:
- [ ] Add confidence intervals for frame importance
- [ ] Show temporal evolution of attention over video
- [ ] Include comparison with baseline methods
- [ ] Add anatomical annotations
- [ ] Generate interactive HTML visualizations
- [ ] Support video output (animated attention)

## Citation

If you use this visualization code, please cite:

```bibtex
@article{yourpaper2026,
    title={Spatio-Temporal Evidence Distillation: Interpretable Ocular Ultrasound Diagnosis 
           via Intrinsic Attention-Guided Vision-Language Models},
    author={Your Name et al.},
    journal={NeurIPS},
    year={2026}
}
```

## Contact

For questions or issues, please open an issue on GitHub or contact the authors.
