import torch
import base64
import uuid
import io
from torch import autocast
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

def stable_diff(prompt, num_inference_steps, guidance_scale, h, w):
    try:
        if not prompt.strip():
            return {"error": "Invalid prompt. The prompt cannot be empty."}
        if h <= 0 or w <= 0:
            return {"error": "Invalid dimensions. Height and width must be greater than zero."}

        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
            pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            pipe = pipe.to(device)
        except Exception as e:
            return {"error": f"Failed to load Stable Diffusion model: {str(e)}"}

        with autocast("cuda"):
            image = pipe(
                prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=h,
                width=w
            ).images[0]

        uid=str(uuid.uuid4())
        prompt=str(prompt)+uid
        full_len = f"flask_app/outputs/{prompt}.jpg"
        try:
            image.save(full_len, "JPEG")
        except Exception as e:
            return {"error": f"Failed to save image: {str(e)}"}
        try:
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            generated_image_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        except Exception as e:
            return {"error": f"Failed to convert image to base64 string: {str(e)}"}

        return generated_image_b64
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}
