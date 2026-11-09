import argparse
from pathlib import Path

import numpy as np
import openslide
from PIL import Image


def keep_patch(image, patch_size):
    gray = np.asarray(image.convert("L"))
    tissue_fraction = np.mean(gray < 200)
    dark_fraction = np.mean(gray < 30)
    return tissue_fraction > 0.5 and dark_fraction < 0.3


def extract_slide(slide_path, output_dir, patch_size):
    slide_id = slide_path.stem
    patch_dir = output_dir / slide_id
    patch_dir.mkdir(parents=True, exist_ok=True)
    coordinates = []
    slide = openslide.OpenSlide(str(slide_path))
    width, height = slide.level_dimensions[0]
    patch_index = 0

    try:
        for y in range(0, height - patch_size + 1, patch_size):
            for x in range(0, width - patch_size + 1, patch_size):
                patch = slide.read_region((x, y), 0, (patch_size, patch_size)).convert("RGB")
                if keep_patch(patch, patch_size):
                    patch.save(patch_dir / f"{slide_id}_{patch_index:06d}.jpg", quality=95)
                    coordinates.append((x, y))
                    patch_index += 1
    finally:
        slide.close()

    np.save(output_dir / f"{slide_id}_coordinates.npy", np.asarray(coordinates, dtype=np.int64))
    print(f"{slide_id}: saved {patch_index} patches")


def main():
    parser = argparse.ArgumentParser(description="Extract tissue patches from SVS slides.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--patch-size", type=int, default=512)
    args = parser.parse_args()
    if args.patch_size <= 0:
        parser.error("--patch-size must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    slides = sorted(args.input_dir.rglob("*.svs"))
    if not slides:
        parser.error(f"No .svs files found under {args.input_dir}")
    for slide_path in slides:
        extract_slide(slide_path, args.output_dir, args.patch_size)


if __name__ == "__main__":
    main()
