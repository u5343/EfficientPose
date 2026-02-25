from pathlib import Path

import cv2
import torch
import yaml
from torch.utils.data import Dataset


class LinemodDataset(Dataset):
    def __init__(self, root_dir, list_file=None, ids=None, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform

        if ids is not None:
            self.ids = [str(x) for x in ids]
        elif list_file:
            with open(list_file, "r", encoding="utf-8") as f:
                self.ids = [line.strip() for line in f.readlines() if line.strip()]
        else:
            rgb_dir = self.root_dir / "rgb"
            if not rgb_dir.exists():
                raise FileNotFoundError(f"rgb folder not found: {rgb_dir}")
            self.ids = sorted(p.stem for p in rgb_dir.glob("*.png"))

        gt_path = self.root_dir / "gt.yml"
        if not gt_path.exists():
            gt_path = self.root_dir.parent / "gt.yml"

        info_path = self.root_dir / "info.yml"
        if not info_path.exists():
            info_path = self.root_dir.parent / "info.yml"

        if not gt_path.exists():
            raise FileNotFoundError(f"gt.yml not found under {self.root_dir} or {self.root_dir.parent}")
        if not info_path.exists():
            raise FileNotFoundError(f"info.yml not found under {self.root_dir} or {self.root_dir.parent}")
        if len(self.ids) == 0:
            raise ValueError("No frame ids found for dataset")

        with open(gt_path, "r", encoding="utf-8") as f:
            self.gt_data = yaml.load(f, Loader=yaml.FullLoader)
        with open(info_path, "r", encoding="utf-8") as f:
            self.info_data = yaml.load(f, Loader=yaml.FullLoader)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id_str = self.ids[idx]
        id_int = int(id_str)

        img_path = self.root_dir / "rgb" / f"{id_str}.png"
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        anno = self.gt_data[id_int][0]
        rotation = torch.tensor(anno["cam_R_m2c"], dtype=torch.float32).view(3, 3)
        translation = torch.tensor(anno["cam_t_m2c"], dtype=torch.float32) / 1000.0

        cam_k = torch.tensor(self.info_data[id_int]["cam_K"], dtype=torch.float32).view(3, 3)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        return {
            "image": image,
            "gt_R": rotation,
            "gt_t": translation,
            "cam_K": cam_k,
        }
