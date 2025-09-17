# Project Directory Structure
stable-diffusion-runpod/
├── app.py                 # Main FastAPI application
├── requirements.txt       # Python dependencies
├── setup.sh              # Setup script for RunPod
├── run.sh                # Production run script
├── config.py             # Configuration settings
├── test_api.py           # Local testing script
├── README.md             # Documentation
└── .gitignore           # Git ignore file

# config.py - Configuration settings
import os
from typing import Optional

class Config:
    # Model settings
    MODEL_ID = "runwayml/stable-diffusion-v1-5"
    TORCH_DTYPE = "float16"
    DEVICE = "cuda"

    # Server settings
    HOST = "0.0.0.0"
    PORT = 8000

    # Generation defaults
    DEFAULT_STEPS = 20
    DEFAULT_GUIDANCE = 7.5
    DEFAULT_WIDTH = 512
    DEFAULT_HEIGHT = 512
    MAX_STEPS = 50
    MAX_GUIDANCE = 20.0

    # Performance settings
    ENABLE_ATTENTION_SLICING = True
    ENABLE_CPU_OFFLOAD = True
    ENABLE_XFORMERS = True

    # Safety settings
    DISABLE_SAFETY_CHECKER = True

    @classmethod
    def get_model_kwargs(cls):
        return {
            "torch_dtype": getattr(torch, cls.TORCH_DTYPE),
            "safety_checker": None if cls.DISABLE_SAFETY_CHECKER else "default",
            "requires_safety_checker": not cls.DISABLE_SAFETY_CHECKER
        }

# test_api.py - Local testing script
import requests
import base64
from PIL import Image
import io
import time

def test_api(base_url: str = "http://localhost:8000"):
    """Test the Stable Diffusion API"""

    print(f"Testing API at {base_url}")

    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            health = response.json()
            print(f"Health check: {health}")
        else:
            print(f"Health check failed: {response.status_code}")
            return
    except Exception as e:
        print(f"Cannot connect to API: {e}")
        return

    # Test image generation
    test_prompts = [
        "a cute cat wearing a wizard hat",
        "a beautiful mountain landscape at sunset",
        "a cyberpunk city with neon lights"
    ]

    for i, prompt in enumerate(test_prompts):
        print(f"\nTesting prompt {i+1}: {prompt}")

        payload = {
            "prompt": prompt,
            "negative_prompt": "blurry, low quality",
            "steps": 15,  # Faster for testing
            "guidance": 7.5
        }

        try:
            start_time = time.time()
            response = requests.post(f"{base_url}/generate", json=payload, timeout=120)

            if response.status_code == 200:
                result = response.json()

                # Save image
                if "image_base64" in result:
                    img_data = base64.b64decode(result["image_base64"])
                    image = Image.open(io.BytesIO(img_data))
                    filename = f"test_{i+1}.png"
                    image.save(filename)

                    print(f"Image saved as {filename}")
                    print(f"Generation time: {result.get('generation_time', 'N/A')}s")
                    print(f"Total time: {time.time() - start_time:.1f}s")
                else:
                    print("No image data in response")
            else:
                print(f"Request failed: {response.status_code}")
                print(response.text)

        except Exception as e:
            print(f"Request error: {e}")

if __name__ == "__main__":
    # Test local server
    test_api("http://localhost:8000")

    # Test RunPod server (uncomment and update URL)
    # test_api("https://your-pod-id-8000.proxy.runpod.net")

# README.md content
"""
# Stable Diffusion API for RunPod

A FastAPI-based Stable Diffusion image generation service optimized for RunPod deployment.

## Features

- Web interface for easy testing
- REST API for programmatic access
- Optimized for GPU memory usage
- Health monitoring endpoints
- Production-ready logging

## Local Development

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run locally:
   ```bash
   python app.py
   ```

3. Test the API:
   ```bash
   python test_api.py
   ```

## RunPod Deployment

1. Upload files to RunPod:
   ```bash
   scp -P [port] -r . root@[ip]:/workspace/
   ```

2. Run setup:
   ```bash
   chmod +x setup.sh run.sh
   ./setup.sh
   ```

3. Start server:
   ```bash
   ./run.sh
   ```

## API Endpoints

- `GET /` - Web interface
- `POST /generate` - Generate image
- `GET /health` - Health check
- `GET /docs` - API documentation

## Configuration

Edit `config.py` to modify:
- Model settings
- Generation parameters
- Performance optimizations

## Usage Examples

### Web Interface
Visit the root URL in your browser for a user-friendly interface.

### API Call
```python
import requests

response = requests.post("http://your-api/generate", json={
    "prompt": "a beautiful landscape",
    "steps": 20,
    "guidance": 7.5
})

result = response.json()
image_base64 = result["image_base64"]
```

## Troubleshooting

- Check GPU memory: `nvidia-smi`
- View logs: `tail -f logs/server.log`
- Restart: `screen -r sdapi` then Ctrl+C and `./run.sh`
"""

# .gitignore content
"""
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Logs
logs/
*.log

# Generated images
outputs/
*.png
*.jpg
*.jpeg

# Model cache
models/
.cache/

# Environment variables
.env
.env.local

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/
"""
