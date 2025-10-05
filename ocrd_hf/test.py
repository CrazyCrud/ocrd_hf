#!/usr/bin/env python3

import argparse
import sys
from PIL import Image
import torch
from adapters import build_adapter

def main():
    parser = argparse.ArgumentParser(
        description="Test OCR-D HF adapter: recognize a single line image"
    )
    parser.add_argument("image_path", help="Path to the line image (PNG, JPG, etc.)")
    parser.add_argument(
        "-m", "--model-id",
        default="microsoft/trocr-base-printed",
        help="HuggingFace model id or local path"
    )
    parser.add_argument(
        "-d", "--device",
        default=None,
        help="Device to run on, e.g. 'cpu' or 'cuda:0'. If omitted, auto-select (cuda if available)."
    )
    parser.add_argument(
        "--fp16", action="store_true",
        help="Use half precision (fp16) if supported"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=None,
        help="Maximum tokens to generate"
    )

    args = parser.parse_args()

    # Decide device
    if args.device:
        device_str = args.device
    else:
        # auto
        device_str = "cuda:0" if torch.cuda.is_available() else "cpu"

    print(f"Using device = {device_str}")

    try:
        adapter = build_adapter(
            model_id=args.model_id,
            device=device_str,
            fp16=args.fp16,
            max_new_tokens=args.max_new_tokens
        )
    except Exception as e:
        print("Failed to build adapter:", e)
        sys.exit(1)

    # Load the image
    try:
        img = Image.open(args.image_path).convert("RGB")
    except Exception as e:
        print("Error loading image:", e)
        sys.exit(1)

    # Recognize
    try:
        pixel_values = adapter.preprocess([img])
        outputs = adapter.generate(pixel_values)
        texts = adapter.decode(outputs)
    except Exception as e:
        print("Inference error:", e)
        sys.exit(1)

    if texts:
        print("Predicted:", texts[0])
    else:
        print("No text predicted.")

if __name__ == "__main__":
    main()
