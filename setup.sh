#!/bin/bash
# setup.sh - FLUX Setup Script with HF Authentication and Auto-start

echo "Setting up FLUX Image Generation API on RunPod..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Update system
print_status "Updating system packages..."
apt update && apt upgrade -y

# Install system dependencies
print_status "Installing system dependencies..."
apt install -y git wget curl screen htop nano

# Navigate to workspace
cd /workspace

# Install/upgrade Python dependencies
print_status "Installing Python dependencies..."
pip install --upgrade pip

# Install requirements
pip install --no-cache-dir \
    torch>=2.0.0 \
    torchvision>=0.15.0 \
    diffusers>=0.30.0 \
    transformers>=4.35.0 \
    accelerate>=0.24.0 \
    safetensors>=0.4.0 \
    xformers>=0.0.22 \
    fastapi>=0.104.0 \
    uvicorn[standard]>=0.24.0 \
    Pillow>=10.0.0 \
    requests>=2.31.0 \
    pydantic>=2.5.0 \
    huggingface_hub>=0.19.0

# Install latest diffusers from git
print_status "Installing latest diffusers..."
pip install git+https://github.com/huggingface/diffusers

# Create necessary directories
mkdir -p logs outputs models

# Set up Hugging Face authentication
print_status "Setting up Hugging Face authentication..."
print_warning "You need to provide your HF token for FLUX.1-dev access"

# Check if HF_TOKEN is already set
if [ -z "$HF_TOKEN" ]; then
    echo "Please enter your Hugging Face token:"
    read -s HF_TOKEN
    export HF_TOKEN=$HF_TOKEN
    # Save to .bashrc for persistence
    echo "export HF_TOKEN=$HF_TOKEN" >> ~/.bashrc
fi

# Authenticate with Hugging Face
print_status "Authenticating with Hugging Face..."
python3 -c "
from huggingface_hub import login
import os
token = os.getenv('HF_TOKEN')
if token:
    login(token=token)
    print('Successfully authenticated with Hugging Face')
else:
    print('HF_TOKEN not found, authentication may fail')
"

# Test CUDA availability
print_status "Testing CUDA availability..."
python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory // 1024**3}GB')
else:
    print('WARNING: CUDA not available, will use CPU')
"

# Pre-download FLUX model (optional but recommended)
print_status "Pre-downloading FLUX.1-dev model (this may take 10-15 minutes)..."
python3 -c "
try:
    from diffusers import FluxPipeline
    import torch
    print('Downloading FLUX.1-dev model...')
    pipe = FluxPipeline.from_pretrained(
        'black-forest-labs/FLUX.1-dev',
        torch_dtype=torch.bfloat16
    )
    print('FLUX.1-dev model cached successfully!')
except Exception as e:
    print(f'Failed to download FLUX model: {e}')
    print('Model will be downloaded on first use')
"

# Make run script executable if it exists
if [ -f "run.sh" ]; then
    chmod +x run.sh
    print_status "Made run.sh executable"
fi

# Create a simple health check script
cat > health_check.py << 'EOF'
import requests
import sys
import time

def check_health():
    try:
        response = requests.get("http://localhost:8888/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get("model_loaded"):
                print("✅ API is healthy and model is loaded")
                return True
            else:
                print("⚠️ API is running but model not loaded")
                return False
        else:
            print(f"❌ API returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

if __name__ == "__main__":
    if check_health():
        sys.exit(0)
    else:
        sys.exit(1)
EOF

# Create auto-start script
cat > start_api.sh << 'EOF'
#!/bin/bash
echo "Starting FLUX Image Generation API..."

# Set environment variables
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

# Create logs directory
mkdir -p logs

# Start the API in screen session
screen -dmS flux_api bash -c "python3 app.py 2>&1 | tee logs/api.log"

echo "API started in screen session 'flux_api'"
echo "To view logs: screen -r flux_api"
echo "To detach: Ctrl+A, D"
echo "Health check in 30 seconds..."

sleep 30
python3 health_check.py
EOF

chmod +x start_api.sh health_check.py

print_status "Setup complete!"
print_status "Environment variables set:"
echo "  HF_TOKEN: ${HF_TOKEN:0:10}..."

print_status "Next steps:"
echo "1. Make sure your app.py is in /workspace/"
echo "2. Run: ./start_api.sh"
echo "3. Check health: python3 health_check.py"
echo "4. View logs: screen -r flux_api"

# Auto-start if app.py exists
if [ -f "app.py" ]; then
    print_status "Found app.py, starting API automatically..."
    ./start_api.sh
else
    print_warning "app.py not found. Please upload it and run ./start_api.sh"
fi
