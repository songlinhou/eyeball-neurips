"""
Configuration for exp10_explainable_flow_lower_dropout experiment
Extracted from video_classification/run_experiments.py
"""

import os

# Experiment configuration
EXP_NAME = 'exp10_explainable_flow_lower_dropout'

# Model configuration
MODEL_CLASS = 'explainable_flow'
NUM_CLASSES = 2
PRETRAINED = True
DROPOUT = 0.3

# Data configuration
NUM_FRAMES = 32
IMG_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 2

# Training configuration
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

# Loss configuration
LOSS_FUNCTION = 'focal'
FOCAL_GAMMA = 2.0

# Augmentation configuration
USE_AUGMENTATION = True
USE_MIXUP = True
USE_TTA = True

# Paths
DATA_DIR = "../erdes"
SPLITS_DIR = os.path.join(DATA_DIR, "splits", "macula_detached_vs_intact")
SAVE_DIR = "./exp10_results"

# Create directories
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, "models"), exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, "checkpoints"), exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, "logs"), exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, "plots"), exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, "results"), exist_ok=True)

# Training phases
PHASE1_EPOCHS = min(5, NUM_EPOCHS // 3)  # Classifier head training
PHASE2_EPOCHS = NUM_EPOCHS - PHASE1_EPOCHS  # Full fine-tuning

# Early stopping
EARLY_STOPPING_PATIENCE = 7

# Learning rate schedules
PHASE1_LR_MULTIPLIER = 10  # Higher LR for classifier head
SCHEDULER_PATIENCE = 3  # For ReduceLROnPlateau
SCHEDULER_FACTOR = 0.5

# Gradient clipping
GRAD_CLIP_MAX_NORM = 1.0

# Mixup
MIXUP_ALPHA = 0.2
MIXUP_PROBABILITY = 0.5

def get_config():
    """Return configuration as dictionary"""
    return {
        'name': EXP_NAME,
        'model_class': MODEL_CLASS,
        'num_frames': NUM_FRAMES,
        'img_size': IMG_SIZE,
        'batch_size': BATCH_SIZE,
        'num_epochs': NUM_EPOCHS,
        'learning_rate': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'dropout': DROPOUT,
        'loss_function': LOSS_FUNCTION,
        'focal_gamma': FOCAL_GAMMA,
        'use_augmentation': USE_AUGMENTATION,
        'use_mixup': USE_MIXUP,
        'use_tta': USE_TTA
    }
