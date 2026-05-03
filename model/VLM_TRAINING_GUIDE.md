# VLM Training Guide: Checkpoint Management & Resumption

## Overview

The VLM finetuning script now includes automatic checkpoint saving and resumption capabilities to handle training interruptions gracefully.

## Key Features

### 1. **Automatic Checkpoint Saving (Every 100 Steps)**
- Checkpoints are saved every **100 training steps** by default
- Keeps the last **5 checkpoints** to save disk space
- Each checkpoint includes:
  - Model weights (LoRA adapters if using PEFT)
  - Optimizer state
  - Learning rate scheduler state
  - Training step counter
  - Random number generator states

### 2. **Automatic Resumption**
- Training automatically resumes from the **last checkpoint** if interrupted
- No manual intervention required
- Preserves exact training state (optimizer momentum, learning rate, etc.)

### 3. **Checkpoint Directory Structure**
```
vlm_finetuned/
├── checkpoint-100/          # Checkpoint at step 100
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── optimizer.pt
│   ├── scheduler.pt
│   ├── trainer_state.json
│   └── training_args.bin
├── checkpoint-200/          # Checkpoint at step 200
├── checkpoint-300/          # Checkpoint at step 300
├── checkpoint-400/          # Checkpoint at step 400
├── checkpoint-500/          # Checkpoint at step 500 (latest)
├── final_model/             # Final trained model
└── runs/                    # TensorBoard logs
```

## Usage Examples

### Basic Training (Auto-Resume Enabled)
```bash
python model/vlm_finetuning.py \
    --samples_json vlm_training_samples.json \
    --output_dir vlm_finetuned \
    --epochs 10 \
    --batch_size 2 \
    --learning_rate 2e-5 \
    --use_lora \
    --load_in_4bit
```

**What happens:**
1. First run: Starts training from scratch, saves checkpoints every 100 steps
2. If interrupted: Simply re-run the same command
3. Automatic detection: Finds `checkpoint-500` (latest) and resumes from there
4. Continues training: Picks up exactly where it left off

### Custom Save Frequency
```bash
python model/vlm_finetuning.py \
    --samples_json vlm_training_samples.json \
    --output_dir vlm_finetuned \
    --save_steps 50 \
    --eval_steps 50 \
    --epochs 10 \
    --use_lora \
    --load_in_4bit
```
Saves checkpoints every **50 steps** instead of 100.

### Disable Auto-Resume (Start Fresh)
```bash
python model/vlm_finetuning.py \
    --samples_json vlm_training_samples.json \
    --output_dir vlm_finetuned \
    --no_resume \
    --epochs 10 \
    --use_lora \
    --load_in_4bit
```
Ignores existing checkpoints and starts training from scratch.

### Resume from Specific Checkpoint (Manual)
```python
from vlm_finetuning import train_vlm, setup_qwen2vl_for_finetuning, MedicalVLMDataset

# Setup
model, processor = setup_qwen2vl_for_finetuning(use_lora=True, load_in_4bit=True)
train_dataset = MedicalVLMDataset('samples.json', processor)
val_dataset = MedicalVLMDataset('samples.json', processor, contrastive_weight=0.0)

# Train with specific checkpoint
trainer = train_vlm(
    model=model,
    processor=processor,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    output_dir='vlm_finetuned',
    resume_from_checkpoint=True  # Will auto-find latest checkpoint
)
```

## Training Interruption Scenarios

### Scenario 1: System Crash
```bash
# Training starts
python model/vlm_finetuning.py --samples_json data.json --output_dir out --use_lora

# Output:
# Starting training from scratch
# Step 100/1000 - Loss: 2.34 - Checkpoint saved
# Step 200/1000 - Loss: 2.12 - Checkpoint saved
# Step 300/1000 - Loss: 1.98 - Checkpoint saved
# [CRASH - Power outage]

# Simply restart with same command
python model/vlm_finetuning.py --samples_json data.json --output_dir out --use_lora

# Output:
# Found checkpoint: out/checkpoint-300. Resuming training...
# Resuming from checkpoint: out/checkpoint-300
# Step 301/1000 - Loss: 1.96 - Continuing...
```

### Scenario 2: Manual Interruption (Ctrl+C)
```bash
# Training in progress
# Step 450/1000 - Loss: 1.45
# [User presses Ctrl+C]

# Restart
python model/vlm_finetuning.py --samples_json data.json --output_dir out --use_lora

# Output:
# Found checkpoint: out/checkpoint-400. Resuming training...
# (Resumes from step 400, not 450, because checkpoint-450 wasn't saved yet)
```

### Scenario 3: Out of Memory Error
```bash
# Training fails due to OOM
# Step 250/1000 - CUDA out of memory error

# Reduce batch size and resume
python model/vlm_finetuning.py \
    --samples_json data.json \
    --output_dir out \
    --batch_size 1 \
    --use_lora

# Output:
# Found checkpoint: out/checkpoint-200. Resuming training...
# (Continues with smaller batch size)
```

## Monitoring Training Progress

### TensorBoard
```bash
tensorboard --logdir vlm_finetuned/runs
```

View:
- Training loss over time
- Evaluation loss
- Learning rate schedule
- Checkpoints saved

### Check Latest Checkpoint
```bash
ls -lth vlm_finetuned/checkpoint-* | head -1
```

### Training State
```python
import json

# Load trainer state from latest checkpoint
with open('vlm_finetuned/checkpoint-500/trainer_state.json', 'r') as f:
    state = json.load(f)

print(f"Global step: {state['global_step']}")
print(f"Epoch: {state['epoch']}")
print(f"Best metric: {state['best_metric']}")
```

## Best Practices

1. **Regular Checkpointing**: Keep `save_steps=100` for long training runs
2. **Disk Space**: Monitor checkpoint directory size (each checkpoint ~1-2GB for 7B model)
3. **Backup**: Periodically backup checkpoint directories to external storage
4. **Validation**: Check `eval_loss` in TensorBoard to ensure training is progressing
5. **Final Model**: Always use `final_model/` directory for inference, not checkpoints

## Troubleshooting

### "No checkpoint found" when checkpoints exist
**Cause**: Checkpoint directory structure is corrupted
**Solution**: Check that checkpoint folders contain all required files

### Training doesn't resume from expected step
**Cause**: Checkpoint wasn't fully saved before interruption
**Solution**: Training resumes from last **complete** checkpoint

### Out of disk space
**Cause**: Too many checkpoints saved
**Solution**: Reduce `save_total_limit` or increase `save_steps`

```python
# In train_vlm function, modify:
save_total_limit=3,  # Keep only last 3 checkpoints
save_steps=200,      # Save less frequently
```

## Advanced: Custom Checkpoint Strategy

```python
from transformers import TrainerCallback

class CustomCheckpointCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        print(f"Checkpoint saved at step {state.global_step}")
        # Add custom logic (e.g., upload to cloud storage)

# Add to trainer
trainer = Trainer(
    model=model,
    args=training_args,
    callbacks=[CustomCheckpointCallback()]
)
```

## Summary

- ✅ **Automatic checkpoint saving every 100 steps**
- ✅ **Automatic resumption from last checkpoint**
- ✅ **Keeps last 5 checkpoints to save space**
- ✅ **Preserves exact training state**
- ✅ **No manual intervention needed**
- ✅ **Configurable via command-line arguments**

Training interruptions are now handled seamlessly!
