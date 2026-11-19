"""Experiment 7: Stable Diffusion 1.5 inference (text-to-image and image-to-image) on a 6 GB GPU.

  python stable_diffusion.py --prompt "a lighthouse at sunset, oil painting"
  python stable_diffusion.py --prompt "same scene in winter" --image runs/txt2img_0.png --strength 0.6
"""
import argparse
import time
from pathlib import Path

import torch
from diffusers import AutoPipelineForImage2Image, StableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
from PIL import Image

RUNS = Path(__file__).resolve().parent / "runs"
MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", required=True)
    p.add_argument("--negative", default="blurry, low quality, deformed")
    p.add_argument("--image", help="init image -> image-to-image")
    p.add_argument("--strength", type=float, default=0.65, help="img2img: how far to move from the init image")
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--guidance", type=float, default=7.5)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", default=MODEL)
    a = p.parse_args()

    pipe = StableDiffusionPipeline.from_pretrained(a.model, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_attention_slicing()          # keeps peak VRAM well under 6 GB
    pipe.enable_model_cpu_offload()
    if a.image:
        pipe = AutoPipelineForImage2Image.from_pipe(pipe)
    gen = torch.Generator("cuda").manual_seed(a.seed)
    kw = dict(prompt=a.prompt, negative_prompt=a.negative, num_inference_steps=a.steps,
              guidance_scale=a.guidance, num_images_per_prompt=a.n, generator=gen)
    if a.image:
        kw.update(image=Image.open(a.image).convert("RGB").resize((a.size, a.size)), strength=a.strength)
    else:
        kw.update(height=a.size, width=a.size)

    t = time.time()
    out = pipe(**kw).images
    RUNS.mkdir(exist_ok=True)
    tag = "img2img" if a.image else "txt2img"
    for i, im in enumerate(out):
        path = RUNS / f"{tag}_{i}.png"
        im.save(path)
        print("saved", path)
    print(f"{time.time() - t:.1f}s | peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")


if __name__ == "__main__":
    main()
