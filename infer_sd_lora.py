import argparse, os, torch
from pathlib import Path
from diffusers import StableDiffusionPipeline
from PIL import Image
from tqdm import trange

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lora_dir", type=str, required=True)
    p.add_argument("--model_id", type=str, default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--negative", type=str, default="")
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--outdir", type=str, default="./samples")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance", type=float, default=7.5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(args.model_id, torch_dtype=dtype)
    pipe.to(device)
    pipe.unet.load_attn_procs(args.lora_dir)

    g = torch.Generator(device=device)
    if args.seed >= 0:
        g.manual_seed(args.seed)

    os.makedirs(args.outdir, exist_ok=True)

    for i in trange(args.n, desc="Generating"):
        img = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative if args.negative else None,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            generator=g
        ).images[0]
        out = Path(args.outdir) / f"sample_{i:02d}.png"
        img.save(out)

    print(f"Saved {args.n} images to {args.outdir}")

if __name__ == "__main__":
    main()
