from __future__ import annotations

import csv
from pathlib import Path

from app_ml.config.settings import (
    RAW_DIR,
    DEGRADED_DIR,
    METADATA_DIR,
    IMAGE_EXTENSIONS,
    MAX_IMAGES,
    VERSIONS_PER_IMAGE,
    RANDOM_SEED,
)
from app_ml.utils.image_ops import (
    load_image,
    save_image,
    degrade_image,
    make_correction_targets,
    list_images,
    set_seed,
    compute_mean_brightness,
    compute_contrast_std,
    compute_mean_saturation,
    sample_degradation_for_image,
)


def relative_to_project(path: Path) -> str:
    return str(path).replace("\\", "/")


def build_dataset() -> None:
    set_seed(RANDOM_SEED)

    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    DEGRADED_DIR.mkdir(parents=True, exist_ok=True)

    raw_images = list_images(RAW_DIR, IMAGE_EXTENSIONS)

    if not raw_images:
        print(f"[ERROR] В папке {RAW_DIR} не найдено изображений.")
        return

    raw_images = raw_images[:MAX_IMAGES]

    csv_path = METADATA_DIR / "dataset.csv"

    fieldnames = [
        "sample_id",
        "raw_path",
        "degraded_path",
        "raw_mean_brightness",
        "raw_contrast_std",
        "raw_mean_saturation",
        "raw_state",
        "degradation_profile",
        "brightness_deg",
        "contrast_deg",
        "saturation_deg",
        "target_brightness",
        "target_contrast",
        "target_saturation",
    ]

    rows_written = 0

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for image_idx, raw_path in enumerate(raw_images):
            try:
                image = load_image(raw_path)
            except Exception as e:
                print(f"[WARN] Не удалось открыть {raw_path}: {e}")
                continue

            raw_stem = raw_path.stem

            raw_mean_brightness = compute_mean_brightness(image)
            raw_contrast_std = compute_contrast_std(image)
            raw_mean_saturation = compute_mean_saturation(image)

            for version_idx in range(VERSIONS_PER_IMAGE):
                (
                    raw_state,
                    degradation_profile,
                    brightness_deg,
                    contrast_deg,
                    saturation_deg,
                ) = sample_degradation_for_image(
                    mean_brightness=raw_mean_brightness,
                    contrast_std=raw_contrast_std,
                    mean_saturation=raw_mean_saturation,
                )

                degraded = degrade_image(
                    image=image,
                    brightness_delta=brightness_deg,
                    contrast_factor=contrast_deg,
                    saturation_factor=saturation_deg,
                )

                target_brightness, target_contrast, target_saturation = make_correction_targets(
                    brightness_deg,
                    contrast_deg,
                    saturation_deg,
                )

                sample_id = f"{image_idx:05d}_{version_idx:02d}"
                degraded_name = f"{raw_stem}_{sample_id}.jpg"
                degraded_path = DEGRADED_DIR / degraded_name

                save_image(degraded, degraded_path)

                writer.writerow(
                    {
                        "sample_id": sample_id,
                        "raw_path": relative_to_project(raw_path),
                        "degraded_path": relative_to_project(degraded_path),
                        "raw_mean_brightness": round(raw_mean_brightness, 6),
                        "raw_contrast_std": round(raw_contrast_std, 6),
                        "raw_mean_saturation": round(raw_mean_saturation, 6),
                        "raw_state": raw_state,
                        "degradation_profile": degradation_profile,
                        "brightness_deg": round(brightness_deg, 6),
                        "contrast_deg": round(contrast_deg, 6),
                        "saturation_deg": round(saturation_deg, 6),
                        "target_brightness": round(target_brightness, 6),
                        "target_contrast": round(target_contrast, 6),
                        "target_saturation": round(target_saturation, 6),
                    }
                )

                rows_written += 1

    print(f"[OK] Готово. Сохранено строк: {rows_written}")
    print(f"[OK] CSV: {csv_path}")


if __name__ == "__main__":
    build_dataset()