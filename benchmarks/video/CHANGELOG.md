# Changelog

## [1.0.2] - 2026-05-03

### Fixed
- **C3D Model**: Fixed batch size mismatch error in forward pass
  - Added `AdaptiveAvgPool3d` layer to ensure consistent feature dimensions
  - Changed `x.view(-1, 8192)` to `x.view(x.size(0), -1)` for proper batch handling
  - Model now works with variable input sizes and batch sizes

- **X3D Model**: Fixed AttributeError when accessing S3D classifier structure
  - Properly handles S3D's classifier architecture (Sequential with Conv3d)
  - Extracts feature dimension from Conv3d `in_channels` instead of `in_features`
  - Added adaptive pooling for S3D backbone to ensure consistent feature size
  - Model now correctly works with both S3D and R3D backbones

## [1.0.1] - 2026-05-02

### Fixed
- **MViT Model**: Fixed tensor size mismatch error when using frame counts other than 16
  - Added automatic temporal sampling/pooling to match pretrained model's expected input (16 frames)
  - Model now works with any frame count (e.g., 32 frames)
  - Sampling uses linear interpolation to select representative frames
  
- **VideoMAE Model**: Fixed tensor size mismatch error (same issue as MViT)
  - Added automatic temporal sampling/pooling to 16 frames
  - Compatible with variable frame counts

### Changed
- Updated `run_quick_test.sh` to accept optional output directory parameter
  - Usage: `./run_quick_test.sh [output_directory]`
  - Default: `./test_results`
  
- Updated `run_full_benchmark.sh` to accept optional output directory parameter
  - Usage: `./run_full_benchmark.sh [output_directory]`
  - Default: `./results_YYYYMMDD_HHMMSS` (timestamped)

### Documentation
- Added notes in README.md about MViT and VideoMAE frame sampling behavior
- Created USAGE_EXAMPLES.md with examples of using custom output directories
- Created CHANGELOG.md to track changes

## [1.0.0] - 2026-05-02

### Added
- Initial release of video classification benchmark
- 8 state-of-the-art video classification models:
  - ResNet3D (baseline)
  - I3D (Inflated 3D ConvNet)
  - SlowFast (dual-pathway network)
  - X3D (efficient network)
  - MViT (vision transformer)
  - VideoMAE (masked autoencoder)
  - TimeSformer (space-time attention)
  - C3D (classic 3D CNN)
- Complete training and evaluation pipeline
- Comprehensive visualization and comparison tools
- Detailed documentation and quick start guides
- Shell scripts for easy execution
