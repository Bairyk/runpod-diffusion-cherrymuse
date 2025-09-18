from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import base64
import io
from PIL import Image
import uvicorn
from pydantic import BaseModel
from typing import Optional
import gc
import time
from diffusers import FluxPipeline
from huggingface_hub import login
import torch

app = FastAPI(title="Stable Diffusion API")
pipe = None

class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = ""  # FLUX doesn't need negative prompts
    steps: Optional[int] = 28  # FLUX.1-dev optimal steps
    guidance: Optional[float] = 3.5  # FLUX guidance scale
    width: Optional[int] = 1024
    height: Optional[int] = 1024
    seed: Optional[int] = None

# Update generation call
@app.post("/generate")
async def generate_image(request: GenerateRequest):
    # ... existing code ...
    
    # FLUX generation
    image = pipe(
        prompt=request.prompt,
        guidance_scale=request.guidance,
        num_inference_steps=request.steps,
        width=request.width,
        height=request.height,
        generator=generator
    ).images[0]

def load_model():
    global pipe
    if pipe is not None:
        return pipe
    
    print("Loading FLUX.1-dev...")
    
    # Authenticate with HF (if not already done)
    try:
        # This will use your saved token
        login()
    except:
        print("HF authentication may be needed")
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    try:
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.bfloat16
        )
        pipe = pipe.to(device)
        pipe.enable_model_cpu_offload()  # For your 8GB VRAM
        
        print("FLUX.1-dev loaded successfully!")
        return pipe
        
    except Exception as e:
        print(f"Failed to load FLUX.1-dev: {e}")
        # Fallback to SD 1.5
        from diffusers import StableDiffusionPipeline
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", 
            torch_dtype=torch.float16
        )
        pipe = pipe.to(device)
        print("Fallback to SD 1.5")
        return pipe

@app.on_event("startup")
async def startup():
    load_model()

@app.get("/", response_class=HTMLResponse)
async def web_interface():
    return """
    <!DOCTYPE html>
    <html>
    <head><title>Stable Diffusion Generator</title></head>
    <body style="font-family: Arial; max-width: 700px; margin: 30px auto; padding: 20px; background: #f5f5f5;">
        <div style="background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
            <h1 style="color: #333; text-align: center;">Stable Diffusion Generator</h1>
            <form id="form">
                <div style="margin: 15px 0;">
                    <label style="display: block; margin-bottom: 5px; font-weight: bold;">Prompt:</label>
                    <textarea id="prompt" style="width: 100%; height: 80px; padding: 10px; border: 1px solid #ddd; border-radius: 5px;" placeholder="a beautiful landscape with mountains and lakes"></textarea>
                </div>
                <div style="margin: 15px 0;">
                    <label style="display: block; margin-bottom: 5px; font-weight: bold;">Negative Prompt:</label>
                    <input id="negative_prompt" style="width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 5px;" placeholder="blurry, low quality, distorted">
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin: 15px 0;">
                    <div>
                        <label style="display: block; margin-bottom: 5px; font-weight: bold;">Steps:</label>
                        <input id="steps" type="number" value="20" min="1" max="50" style="width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 5px;">
                    </div>
                    <div>
                        <label style="display: block; margin-bottom: 5px; font-weight: bold;">Guidance:</label>
                        <input id="guidance" type="number" value="7.5" step="0.5" min="1" max="20" style="width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 5px;">
                    </div>
                </div>
                <button type="submit" style="width: 100%; padding: 15px; background: #007bff; color: white; border: none; border-radius: 5px; font-size: 16px; cursor: pointer;">Generate Image</button>
            </form>
            <div id="result" style="margin-top: 30px;"></div>
        </div>

        <script>
        document.getElementById('form').addEventListener('submit', async function(e) {
            e.preventDefault();
            const result = document.getElementById('result');
            const button = document.querySelector('button');

            button.disabled = true;
            button.textContent = 'Generating...';
            result.innerHTML = '<div style="text-align: center; padding: 20px; color: #666;">Generating image... This may take 30-60 seconds.</div>';

            const data = {
                prompt: document.getElementById('prompt').value,
                negative_prompt: document.getElementById('negative_prompt').value,
                steps: parseInt(document.getElementById('steps').value),
                guidance: parseFloat(document.getElementById('guidance').value)
            };

            try {
                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(data)
                });

                const jsonResult = await response.json();
                if (response.ok) {
                    result.innerHTML = `
                        <div style="text-align: center;">
                            <h3>Generated Image</h3>
                            <img src="data:image/png;base64,${jsonResult.image_base64}" style="max-width: 100%; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.2);">
                            <div style="margin-top: 15px; padding: 10px; background: #f8f9fa; border-radius: 5px; font-size: 14px;">
                                <strong>Prompt:</strong> ${jsonResult.prompt}<br>
                                <strong>Generation Time:</strong> ${jsonResult.generation_time?.toFixed(1)}s
                            </div>
                        </div>
                    `;
                } else {
                    result.innerHTML = `<div style="color: red; text-align: center; padding: 20px;">Error: ${jsonResult.error}</div>`;
                }
            } catch (error) {
                result.innerHTML = `<div style="color: red; text-align: center; padding: 20px;">Network Error: ${error.message}</div>`;
            }

            button.disabled = false;
            button.textContent = 'Generate Image';
        });
        </script>
    </body>
    </html>
    """

@app.post("/generate")
async def generate_image(request: GenerateRequest):
    global pipe
    if pipe is None:
        load_model()

    try:
        start_time = time.time()

        generator = None
        if request.seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(request.seed)

        print(f"Generating: {request.prompt[:50]}...")

        with torch.autocast("cuda"):
            result = pipe(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt if request.negative_prompt else None,
                num_inference_steps=request.steps,
                guidance_scale=request.guidance,
                generator=generator
            )

        generation_time = time.time() - start_time
        image = result.images[0]

        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode()

        torch.cuda.empty_cache()
        gc.collect()

        print(f"Generated in {generation_time:.1f}s")

        return {
            "image_base64": img_base64,
            "prompt": request.prompt,
            "generation_time": generation_time
        }

    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e)}

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": pipe is not None,
        "gpu_available": torch.cuda.is_available()
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
