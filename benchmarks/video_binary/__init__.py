"""
Video Classification Benchmark for Medical Videos

This package provides a comprehensive benchmark for comparing state-of-the-art
video classification methods on medical video data (ERDES dataset).

Modules:
    video_models: Implementation of 8 video classification models
    train_benchmark: Training and evaluation pipeline
    compare_results: Visualization and comparison tools

Models included:
    - ResNet3D: Standard 3D ResNet baseline
    - I3D: Inflated 3D ConvNet (CVPR 2017)
    - SlowFast: Dual-pathway network (ICCV 2019)
    - X3D: Efficient video network (CVPR 2020)
    - MViT: Multiscale Vision Transformer (ICCV 2021)
    - VideoMAE: Masked Autoencoder (NeurIPS 2022)
    - TimeSformer: Space-time attention (ICML 2021)
    - C3D: Classic 3D CNN (ICCV 2015)

Usage:
    See README.md and QUICK_START.md for detailed instructions.
"""

__version__ = "1.0.0"
__author__ = "ERDES Benchmark"

from .video_models import (
    I3DModel,
    SlowFastModel,
    X3DModel,
    TimeSformerModel,
    VideoMAEModel,
    MViTModel,
    C3DModel,
    ResNet3DModel,
    get_model
)

__all__ = [
    'I3DModel',
    'SlowFastModel',
    'X3DModel',
    'TimeSformerModel',
    'VideoMAEModel',
    'MViTModel',
    'C3DModel',
    'ResNet3DModel',
    'get_model'
]
