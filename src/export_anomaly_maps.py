"""
Runs inference on every test image for a category and saves heatmaps
as .tiff files in the format expected by MVTec's official evaluation script.

Output structure:
  outputs/anomaly_maps/ategory>/test/<defect_type>/<image_id>.tiff

Usage:
  python -m src.export_anomaly_maps --data_root data --category bottle --device cpu
"""

import argparse
import numpy as np
import tifffile
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from src.dataset import get_transforms
from src.inference import load_model


def export_maps(
    data_root: str,
    category: str,
    output_dir: str = "outputs/anomaly_maps",
    device: str = "cpu"
):
    print(f"\nExporting anomaly maps for: {category}")

    model = load_model(category, device=device)
    model.eval()

    transform = get_transforms(image_size=224)
    test_dir  = Path(data_root) / category / "test"
    save_base = Path(output_dir) / category / "test"

    defect_types = sorted([d for d in test_dir.iterdir() if d.is_dir()])

    for defect_dir in defect_types:
        save_dir = save_base / defect_dir.name
        save_dir.mkdir(parents=True, exist_ok=True)

        image_files = sorted(defect_dir.glob("*.png"))

        for img_path in tqdm(image_files, desc=f"  {defect_dir.name}"):
            image = Image.open(img_path).convert("RGB")
            tensor = transform(image).unsqueeze(0)

            _, heatmap = model.score(tensor)

            # Resize heatmap back to original image dimensions
            # The evaluation script compares against full-resolution ground truth masks
            orig_w, orig_h = image.size
            heatmap_pil = Image.fromarray(heatmap.astype(np.float32))
            heatmap_resized = np.array(
                heatmap_pil.resize((orig_w, orig_h), Image.BILINEAR),
                dtype=np.float32
            )

            # Save as .tiff — MVTec evaluation script reads float32 tiff files
            save_path = save_dir / (img_path.stem + ".tiff")
            tifffile.imwrite(str(save_path), heatmap_resized)

    print(f"\nDone. Maps saved to: {save_base}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export anomaly heatmaps as .tiff files")
    parser.add_argument("--data_root", type=str, required=True, help="Path to MVTec dataset root")
    parser.add_argument("--category",  type=str, default="bottle", help="Category to evaluate")
    parser.add_argument("--output_dir", type=str, default="outputs/anomaly_maps", help="Where to save tiff maps")
    parser.add_argument("--device",    type=str, default="cpu", help="cpu or cuda")
    args = parser.parse_args()

    export_maps(args.data_root, args.category, args.output_dir, device=args.device)