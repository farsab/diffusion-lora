import argparse, os, math, csv, random
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from diffusers.models.attention_processor import LoRAAttnProcessor
from transformers import CLIPTextModel, CLIPTokenizer

# --- Dataset ---
class CaptionImageDataset(Dataset):
    def __init__(self, data_dir: str, captions_csv: str, image_size: int = 512):
        self.data_dir = Path(data_dir) / "images"
        self.samples: List[Tuple[str, str]] = []
        with open(captions_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append((row["image"], row["prompt"]))

        self.preproc = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),  # [0,1]
            transforms.Normalize([0.5], [0.5])  # to [-1,1]
        ])

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        img_name, prompt = self.samples[i]
        img = Image.open(self.data_dir / img_name).convert("RGB")
        pixel_values = self.preproc(img)
        return {"pixel_values": pixel_values, "prompt": prompt}

# --- Utils ---
def set_seed(seed: int):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def add_lora_to_unet(unet: UNet2DConditionModel, rank: int = 8):
    attn_procs = {}
    for name, module in unet.attn_processors.items():
        attn_procs[name] = LoRAAttnProcessor(hidden_size=module.hidden_size, rank=rank)
    unet.set_attn_processor(attn_procs)
    # return only trainable params
    lora_params = []
    for _, p in unet.attn_processors.items():
        lora_params += list(p.parameters())
    return lora_params

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--captions", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_id", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load base model parts ---
    tokenizer = CLIPTokenizer.from_pretrained(args.model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.model_id, subfolder="text_encoder").to(device)
    vae = AutoencoderKL.from_pretrained(args.model_id, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(args.model_id, subfolder="unet").to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(args.model_id, subfolder="scheduler")

    # Freeze everything except LoRA adapters
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # Inject LoRA into UNet attention modules
    lora_params = add_lora_to_unet(unet, rank=args.rank)
    optimizer = torch.optim.AdamW(lora_params, lr=args.lr)

    # Data
    ds = CaptionImageDataset(args.data_dir, args.captions, image_size=args.image_size)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, pin_memory=True, drop_last=True)

    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
    latents_scale = 0.18215  # SD latent scaling

    global_step = 0
    steps_per_epoch = math.ceil(len(ds) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs

    print(f"Train samples: {len(ds)} | Steps/Epoch: {steps_per_epoch} | Total Steps: {total_steps}")
    unet.train()

    for epoch in range(args.epochs):
        pbar = tqdm(dl, desc=f"Epoch {epoch+1}/{args.epochs}", leave=True)
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(pbar):
            with torch.no_grad():
                # Tokenize prompts
                tokens = tokenizer(batch["prompt"], padding="max_length", truncation=True,
                                   max_length=tokenizer.model_max_length, return_tensors="pt")
                input_ids = tokens.input_ids.to(device)
                enc_out = text_encoder(input_ids)
                enc_hidden = enc_out.last_hidden_state

                # Encode images to latents
                pixel_values = batch["pixel_values"].to(device, dtype=torch.float32)
                latents = vae.encode(pixel_values).latent_dist.sample() * latents_scale

                # Sample noise & timesteps
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],), device=device, dtype=torch.long
                )
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            with torch.cuda.amp.autocast(enabled=args.fp16):
                # Predict noise
                pred = unet(noisy_latents, timesteps, encoder_hidden_states=enc_hidden).sample
                loss = F.mse_loss(pred.float(), noise.float(), reduction="mean") / args.grad_accum

            scaler.scale(loss).backward()

            if (step + 1) % args.grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            pbar.set_postfix({"loss": f"{loss.item()*args.grad_accum:.4f}", "step": global_step})

        # Save LoRA adapters each epoch
        save_dir = Path(args.output_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        unet.save_attn_procs(save_dir)
        torch.save({"epoch": epoch+1, "global_step": global_step}, save_dir / "train_state.pt")
        print(f"Saved LoRA weights to {save_dir}")

    print("Done.")

if __name__ == "__main__":
    main()
