"""
Multi-class video classification benchmark for hierarchical diagnosis.

This benchmark trains and evaluates video classification models on:
- Diagnostic class: Primary diagnosis (non_rd, rd)
- Subtype: Subtype classification (macula_detached, macula_intact, normal, pvd)
"""

from .multiclass_models import get_multiclass_model
from .multiclass_dataset import create_multiclass_dataloaders
from .prepare_splits import prepare_multiclass_splits

__all__ = [
    'get_multiclass_model',
    'create_multiclass_dataloaders',
    'prepare_multiclass_splits'
]
