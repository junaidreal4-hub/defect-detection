import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

from src.dataset import MVTecDataset
from src.patchcore import PatchCore


def train(
    data_root: str,
    category: str,
    output_dir: str = "outputs/memory_bank",
    image_size: int = 256,
    batch_size: int = 8,
    coreset_ratio: float = 0.5,
    device: str = "cpu",
):
    print(f"\nTraining PatchCore on category: {category}")

    dataset = MVTecDataset(data_root, category, train=True, image_size=image_size)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = PatchCore(device=device, coreset_ratio=coreset_ratio)
    model.eval()

    all_patches = []

    for images, _, _ in tqdm(loader, desc="Extracting features"):
        features = model.extract_features(images.to(device))
        patches, H, W = model._reshape_to_patches(features)
        all_patches.append(patches)

    model.build_memory_bank(all_patches)

    # Save the memory bank
    save_path = Path(output_dir) / category
    save_path.mkdir(parents=True, exist_ok=True)
    torch.save(model.memory_bank_tensor, save_path / "memory_bank.pt")
    print(f"Saved memory bank to {save_path}/memory_bank.pt")

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",  type=str, required=True,  help="Path to MVTec dataset root")
    parser.add_argument("--category",   type=str, default="bottle")
    parser.add_argument("--device",     type=str, default="cpu")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    train(args.data_root, args.category, device=args.device, batch_size=args.batch_size)