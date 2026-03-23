from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from app_ml.config.settings import ARTIFACTS_DIR, SPLITS_DIR
from app_ml.models.cnn_regressor import CNNRegressor
from app_ml.utils.image_ops import load_image, save_image, degrade_image


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_project_path(rel_path: str) -> Path:
    return project_root() / rel_path


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def preview_cnn_results(num_samples: int = 10) -> None:
    test_csv_path = SPLITS_DIR / "test.csv"
    model_path = ARTIFACTS_DIR / "checkpoints" / "cnn_regressor_best.pt"
    output_dir = ARTIFACTS_DIR / "reports" / "cnn_preview"

    if not test_csv_path.exists():
        print(f"[ERROR] Не найден test split: {test_csv_path}")
        return

    if not model_path.exists():
        print(f"[ERROR] Не найдена CNN model: {model_path}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(test_csv_path)
    if df.empty:
        print("[ERROR] test.csv пустой.")
        return

    if "preview_path" not in df.columns:
        print("[ERROR] В test.csv нет колонки 'preview_path'.")
        return

    df = df.head(num_samples).copy()

    device = get_device()
    print(f"[INFO] Device: {device}")

    model = CNNRegressor().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    with torch.no_grad():
        for _, row in df.iterrows():
            sample_id = row["sample_id"]

            preview_path = resolve_project_path(row["preview_path"])
            degraded_path = resolve_project_path(row["degraded_path"])
            raw_path = resolve_project_path(row["raw_path"])

            try:
                preview_img = Image.open(preview_path).convert("RGB")
                preview_tensor = transform(preview_img).unsqueeze(0).to(device)
            except Exception as e:
                print(f"[WARN] Не удалось открыть preview {preview_path}: {e}")
                continue

            try:
                degraded_img = load_image(degraded_path)
            except Exception as e:
                print(f"[WARN] Не удалось открыть degraded {degraded_path}: {e}")
                continue

            try:
                raw_img = load_image(raw_path)
            except Exception as e:
                print(f"[WARN] Не удалось открыть raw {raw_path}: {e}")
                raw_img = None

            pred = model(preview_tensor).squeeze(0).cpu().numpy()

            pred_brightness = float(pred[0])
            pred_contrast = float(pred[1])
            pred_saturation = float(pred[2])

            corrected_img = degrade_image(
                image=degraded_img,
                brightness_delta=pred_brightness,
                contrast_factor=pred_contrast,
                saturation_factor=pred_saturation,
            )

            raw_out = output_dir / f"{sample_id}_raw.jpg"
            degraded_out = output_dir / f"{sample_id}_degraded.jpg"
            corrected_out = output_dir / f"{sample_id}_corrected_cnn.jpg"

            save_image(degraded_img, degraded_out)
            save_image(corrected_img, corrected_out)
            if raw_img is not None:
                save_image(raw_img, raw_out)

            print(f"[OK] Saved CNN preview for sample {sample_id}")
            print(
                f"     pred: brightness={pred_brightness:.4f}, "
                f"contrast={pred_contrast:.4f}, saturation={pred_saturation:.4f}"
            )
            print(
                f"     files: {degraded_out.name}, {corrected_out.name}, "
                f"{raw_out.name if raw_img is not None else 'raw not saved'}"
            )

    print(f"\n[OK] CNN preview results saved to: {output_dir}")


if __name__ == "__main__":
    preview_cnn_results(num_samples=10)