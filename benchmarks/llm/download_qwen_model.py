#!/usr/bin/env python3
"""
Script to download vision-language model weights from HuggingFace.
Downloads both Qwen2.5-VL and Llama 3.2 Vision models.
This will cache the models locally for faster subsequent loading.
"""

import os
import sys
from transformers import (
    Qwen2VLForConditionalGeneration, 
    MllamaForConditionalGeneration,
    LlavaNextForConditionalGeneration,
    AutoProcessor,
    LlavaNextProcessor
)
from huggingface_hub import snapshot_download

# Model configurations
MODELS = {
    "qwen": {
        "name": "Qwen/Qwen2.5-VL-7B-Instruct",
        "size": "~15GB",
        "model_class": Qwen2VLForConditionalGeneration,
        "processor_class": AutoProcessor
    },
    "llama": {
        "name": "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "size": "~22GB",
        "model_class": MllamaForConditionalGeneration,
        "processor_class": AutoProcessor
    },
    "llava": {
        "name": "llava-hf/llava-v1.6-mistral-7b-hf",
        "size": "~14GB",
        "model_class": LlavaNextForConditionalGeneration,
        "processor_class": LlavaNextProcessor
    }
}

CACHE_DIR = os.getenv('HF_HOME', os.path.expanduser('~/.cache/huggingface'))

print("=" * 80)
print("Vision-Language Models Downloader")
print("=" * 80)
print(f"Cache directory: {CACHE_DIR}")
print()
print("Models to download:")
for key, info in MODELS.items():
    print(f"  - {info['name']} ({info['size']})")
print("=" * 80)
print()

def download_model(model_key, model_info):
    """Download a specific model."""
    model_name = model_info["name"]
    model_class = model_info["model_class"]
    processor_class = model_info.get("processor_class", AutoProcessor)
    
    print(f"\n{'=' * 80}")
    print(f"Downloading {model_name}")
    print(f"Estimated size: {model_info['size']}")
    print(f"{'=' * 80}\n")
    
    try:
        print(f"[1/2] Downloading processor for {model_key}...")
        processor = processor_class.from_pretrained(
            model_name,
            cache_dir=CACHE_DIR,
            resume_download=True
        )
        print(f"✓ Processor downloaded successfully")
        
        print(f"\n[2/2] Downloading model weights (this may take a while)...")
        model = model_class.from_pretrained(
            model_name,
            cache_dir=CACHE_DIR,
            resume_download=True,
            low_cpu_mem_usage=True
        )
        print(f"✓ Model weights downloaded successfully")
        
        # Clean up to free memory
        del model
        del processor
        
        return True
        
    except Exception as e:
        print(f"✗ Error downloading {model_name}: {e}")
        
        if "llama" in model_key.lower():
            print("\nNote: Llama models require authentication.")
            print("Please run: huggingface-cli login")
            print("And request access at: https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct")
        
        return False

# Download all models
success_count = 0
failed_models = []

for model_key, model_info in MODELS.items():
    if download_model(model_key, model_info):
        success_count += 1
    else:
        failed_models.append(model_info["name"])

# Summary
print("\n" + "=" * 80)
print("DOWNLOAD SUMMARY")
print("=" * 80)
print(f"Successfully downloaded: {success_count}/{len(MODELS)} models")
print(f"Cache location: {CACHE_DIR}")

if failed_models:
    print(f"\nFailed downloads:")
    for model_name in failed_models:
        print(f"  ✗ {model_name}")
    print("\nTroubleshooting tips:")
    print("1. Check your internet connection")
    print("2. Ensure you have enough disk space (~55GB total)")
    print("3. Try running: pip install --upgrade transformers huggingface-hub")
    print("4. For Llama models, run: huggingface-cli login")
else:
    print("\n✓ All models downloaded successfully!")
    print("You can now use the models in your scripts.")

print("=" * 80)
