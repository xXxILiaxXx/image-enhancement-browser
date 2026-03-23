from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from app_ml.config.settings import ARTIFACTS_DIR, SPLITS_DIR
from app_ml.utils.image_ops import load_image, save_image, degrade_image


FEATURE_COLUMNS = [
    "raw_mean_brightness",
    "raw_contrast_std",
    "raw_mean_saturation",
]


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_project_path(rel_path: str) -> Path:
    return project_root() / rel_path


def preview_baseline_results(num_samples: int = 10) -> None:
    test_csv_path = SPLITS_DIR / "test.csv"
    model_path = ARTIFACTS_DIR / "checkpoints" / "baseline_rf.joblib"
    output_dir = ARTIFACTS_DIR / "reports" / "baseline_preview"

    if not test_csv_path.exists():
        print(f"[ERROR] Не найден test split: {test_csv_path}")
        return

    if not model_path.exists():
        print(f"[ERROR] Не найдена baseline model: {model_path}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(test_csv_path)
    if df.empty:
        print("[ERROR] test.csv пустой.")
        return

    df = df.head(num_samples).copy()

    model = joblib.load(model_path)

    x = df[FEATURE_COLUMNS]
    preds = model.predict(x)

    for idx, (_, row) in enumerate(df.iterrows()):
        degraded_path = resolve_project_path(row["degraded_path"])
        raw_path = resolve_project_path(row["raw_path"])

        try:
            degraded_img = load_image(degraded_path)
        except Exception as e:
            print(f"[WARN] Не удалось открыть degraded image {degraded_path}: {e}")
            continue

        try:
            raw_img = load_image(raw_path)
        except Exception as e:
            print(f"[WARN] Не удалось открыть raw image {raw_path}: {e}")
            raw_img = None

        pred_brightness = float(preds[idx][0])
        pred_contrast = float(preds[idx][1])
        pred_saturation = float(preds[idx][2])

        corrected_img = degrade_image(
            image=degraded_img,
            brightness_delta=pred_brightness,
            contrast_factor=pred_contrast,
            saturation_factor=pred_saturation,
        )

        sample_id = row["sample_id"]

        degraded_out = output_dir / f"{sample_id}_degraded.jpg"
        corrected_out = output_dir / f"{sample_id}_corrected.jpg"
        raw_out = output_dir / f"{sample_id}_raw.jpg"

        save_image(degraded_img, degraded_out)
        save_image(corrected_img, corrected_out)
        if raw_img is not None:
            save_image(raw_img, raw_out)

        print(f"[OK] Saved preview for sample {sample_id}")
        print(
            f"     pred: brightness={pred_brightness:.4f}, "
            f"contrast={pred_contrast:.4f}, saturation={pred_saturation:.4f}"
        )
        print(
            f"     files: {degraded_out.name}, {corrected_out.name}, "
            f"{raw_out.name if raw_img is not None else 'raw not saved'}"
        )

    print(f"\n[OK] Preview results saved to: {output_dir}")


if __name__ == "__main__":
    preview_baseline_results(num_samples=10)