#!/bin/bash
# Run the benchmark environment with optimizations for ML workloads

# Remove existing container if it exists
docker rm -f eyeball-benchmarks 2>/dev/null

# Run with optimizations:
# - gpus all: Enable GPU support (if available)
# - shm-size: Increase shared memory for PyTorch (default 64MB is too small)
# - env-file: Load API keys from .env file
# - cache mounts: Mount HuggingFace cache for faster model loading
docker run -it \
    --gpus all \
    --shm-size=16g \
    --env-file llm/.env \
    -v $(pwd):/app \
    -v $(pwd)/../erdes:/erdes \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --name eyeball-benchmarks \
    eyeball-benchmarks \
    bash
