"""Check SAM3 model download progress."""
import os
from pathlib import Path

cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
sam3_path = cache_dir / "models--facebook--sam3"

if sam3_path.exists():
    total_size = 0
    for root, dirs, files in os.walk(sam3_path):
        for file in files:
            file_path = Path(root) / file
            if file_path.exists():
                total_size += file_path.stat().st_size
    
    print(f"Downloaded: {total_size / (1024**3):.2f} GB")
    print(f"Target size: ~3.44 GB")
    print(f"Progress: {(total_size / (3.44 * 1024**3)) * 100:.1f}%")
    
    # Check for model.safetensors specifically
    model_file = None
    for root, dirs, files in os.walk(sam3_path):
        if "model.safetensors" in files:
            model_file = Path(root) / "model.safetensors"
            break
    
    if model_file and model_file.exists():
        model_size = model_file.stat().st_size / (1024**3)
        print(f"\nmodel.safetensors: {model_size:.2f} GB / 3.44 GB")
else:
    print("No download found in cache")

