import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# Standard ImageNet normalization — works well since we use a pretrained backbone
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_transforms(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class MVTecDataset(Dataset):
    """
    Loads MVTec AD images for a single category.

    train=True  → only loads the 'good' training images
    train=False → loads all test images (good + defective)
    """

    def __init__(self, root: str, category: str, train: bool = True, image_size: int = 224):
        self.root = Path(root) / category
        self.train = train
        self.transform = get_transforms(image_size)
        self.image_paths, self.labels = self._load_paths()

    def _load_paths(self):
        paths, labels = [], []

        if self.train:
            good_dir = self.root / "train" / "good"
            for p in sorted(good_dir.glob("*.png")):
                paths.append(p)
                labels.append(0)  # 0 = normal
        else:
            test_dir = self.root / "test"
            for defect_type in sorted(test_dir.iterdir()):
                label = 0 if defect_type.name == "good" else 1
                for p in sorted(defect_type.glob("*.png")):
                    paths.append(p)
                    labels.append(label)

        return paths, labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(image), self.labels[idx], str(self.image_paths[idx])