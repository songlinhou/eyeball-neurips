# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_LAUNCH_BLOCKING=0 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install Python 3.10 and system dependencies (native to Ubuntu 22.04)
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Copy requirements file (if it exists) or install directly
COPY requirements.txt* ./
RUN if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other Python dependencies
RUN pip install --no-cache-dir \
    openai \
    google-generativeai \
    anthropic \
    transformers \
    accelerate \
    qwen-vl-utils \
    pandas \
    opencv-python-headless \
    Pillow \
    tqdm \
    python-dotenv

# Note: Application code should be mounted as a volume when running the container
# Example with GPU: docker run --gpus all -v $(pwd):/app eyeball-benchmarks
# Example without GPU: docker run -v $(pwd):/app eyeball-benchmarks

CMD ["bash"]
