# Diffusion LoRA: Fine-Tuning for Stable Diffusion

Train a LoRA adapter on your own image–text pairs, then generate images with the adapted model.

## 2) Prepare data

Place training images in ./data/images. Create a captions.csv with:
image,prompt
my_img_01.jpg,"a photo of a red vintage car on a city street, cinematic lighting"
...

## 3) Fine-tune (LoRA)
python finetune_sd_lora.py \
  --data_dir ./data \
  --captions ./data/captions.csv \
  --output_dir ./lora_out \
  --model_id runwayml/stable-diffusion-v1-5 \
  --epochs 5 \
  --batch_size 2 \
  --lr 1e-4 \
  --rank 8 \
  --fp16

## 4) Inference with the LoRA
python infer_sd_lora.py \
  --lora_dir ./lora_out \
  --prompt "a photo of a red vintage car on a rainy night, bokeh" \
  --n 4 \
  --outdir ./samples \
  --steps 30 \
  --guidance 7.5
