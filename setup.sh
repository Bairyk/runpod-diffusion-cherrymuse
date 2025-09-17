# setup.sh - Setup script for RunPod deployment
#!/bin/bash

echo "Setting up Stable Diffusion API on RunPod..."

# Update system
apt update && apt upgrade -y

# Install system dependencies
apt install -y git wget curl screen htop

# Create project directory
mkdir -p /workspace/stable-diffusion-api
cd /workspace/stable-diffusion-api

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
mkdir -p logs
mkdir -p outputs
mkdir -p models

# Set permissions
chmod +x run.sh

# Download model (optional - speeds up first run)
echo "Pre-downloading Stable Diffusion model..."
python -c "
from diffusers import StableDiffusionPipeline
import torch
print('Downloading model...')
pipe = StableDiffusionPipeline.from_pretrained(
    'runwayml/stable-diffusion-v1-5',
    torch_dtype=torch.float16
)
print('Model cached successfully!')
"

echo "Setup complete! Run './run.sh' to start the server."
