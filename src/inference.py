import torch
import numpy as np
import cv2
from pathlib import Path
from PIL import Image

from src.dataset import get_transforms
from src.patchcore import PatchCore


def load_model(category: str, memory_bank_dir: str = "outputs/memory_bank", device: str = "cpu") -> PatchCore:
    model = PatchCore(device=device)
    model.eval()

    bank_path = Path(memory_bank_dir) / category / "memory_bank.pt"
    if not bank_path.exists():
        raise FileNotFoundError(f"No memory bank found at {bank_path}. Run train.py first.")

    model.memory_bank_tensor = torch.load(bank_path, map_location=device)
    model.memory_bank = model.memory_bank_tensor.cpu().numpy()
    return model


def predict(model: PatchCore, image_path: str, image_size: int = 224, threshold: float = None):
    """
    Runs inference on a single image.
    Returns anomaly score, binary mask, and overlay visualization.
    """
    original = Image.open(image_path).convert("RGB")
    original_np = np.array(original)

    transform = get_transforms(image_size)
    tensor = transform(original).unsqueeze(0)  # (1, C, H, W)

    score, heatmap = model.score(tensor)

    # Resize heatmap back to original image size for overlay
    heatmap_resized = cv2.resize(heatmap, (original_np.shape[1], original_np.shape[0]))
    heatmap_norm = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min() + 1e-8)
    heatmap_uint8 = (heatmap_norm * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(
        cv2.cvtColor(original_np, cv2.COLOR_RGB2BGR), 0.6,
        heatmap_color, 0.4, 0
    )

    is_defective = score > threshold if threshold else None

    return {
        "score": round(score, 4),
        "is_defective": is_defective,
        "heatmap": heatmap_norm,
        "overlay_bgr": overlay,
    }