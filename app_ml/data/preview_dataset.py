from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from app_ml.config.settings import SPLITS_DIR


TARGET_COLUMNS = [
    "target_brightness",
    "target_contrast",
    "target_saturation",
]


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_project_path(rel_path: str) -> Path:
    return project_root() / rel_path


class PreviewRegressionDataset(Dataset):
    def __init__(self, split_name: str) -> None:
        split_path = SPLITS_DIR / f"{split_name}.csv"
        if not split_path.exists():
            raise FileNotFoundError(f"Не найден split: {split_path}")

        self.df = pd.read_csv(split_path)
        if "preview_path" not in self.df.columns:
            raise ValueError("В split-файле нет колонки 'preview_path'.")

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        image_path = resolve_project_path(row["preview_path"])
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)

        target = torch.tensor(
            [
                row["target_brightness"],
                row["target_contrast"],
                row["target_saturation"],
            ],
            dtype=torch.float32,
        )

        return image_tensor, target