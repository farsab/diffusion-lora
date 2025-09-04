
import argparse, csv, os
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images_dir", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--prefix", type=str, default="")
    args = p.parse_args()

    img_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    images = [f for f in sorted(Path(args.images_dir).iterdir()) if f.suffix.lower() in img_exts]

    os.makedirs(Path(args.out).parent, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "prompt"])
        for im in images:
            # Simple baseline prompt using filename words
            base = " ".join(im.stem.replace("_", " ").replace("-", " ").split())
            w.writerow([im.name, f"{args.prefix}{base}"])

    print(f"Wrote {len(images)} rows to {args.out}")

if __name__ == "__main__":
    main()
