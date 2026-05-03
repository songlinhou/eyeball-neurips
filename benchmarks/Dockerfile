FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file (if it exists) or install directly
COPY requirements.txt* ./
RUN if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

# Install common Python dependencies
RUN pip install --no-cache-dir \
    openai \
    google-generativeai \
    anthropic \
    torch \
    torchvision \
    transformers \
    accelerate \
    qwen-vl-utils \
    pandas \
    opencv-python-headless \
    Pillow \
    tqdm \
    dotenv

# Note: Application code should be mounted as a volume when running the container
# Example: docker run -v $(pwd):/app eyeball-benchmarks

CMD ["bash"]
