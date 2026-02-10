# Multi-stage Dockerfile for VAE training and evaluation
# Base image with CUDA support for GPU training
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set working directory
WORKDIR /workspace

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=America/Los_Angeles

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    unzip \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY ../vae_model.py .
COPY ../dataset.py .
COPY ../train.py .
COPY ../evaluate.py .

# Create output directories
RUN mkdir -p /workspace/outputs /workspace/evaluation /workspace/data

# Set default command to show help
CMD ["python", "train.py", "--help"]

