from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image

from app_ml.config.settings import METADATA_DIR, PREVIEWS_DIR, RANDOM_SEED, PREVIEW_SIZE
from app_ml.utils.image_ops import load_image


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_project_path(rel_path: str) -> Path:
    return project_root() / rel_path


def relative_to_project(path: Path) -> str:
    return str(path.relative_to(project_root())).replace("\\", "/")


def build_preview(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """
    Делает preview фиксированного размера.
    """
    return image.resize(size, Image.Resampling.BILINEAR)


def generate_previews() -> None:
    csv_path = METADATA_DIR / "dataset.csv"

    if not csv_path.exists():
        print(f"[ERROR] Не найден dataset.csv: {csv_path}")
        return

    PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    if df.empty:
        print("[ERROR] dataset.csv пустой.")
        return

    preview_paths = []
    saved_count = 0

    for _, row in df.iterrows():
        sample_id = row["sample_id"]
        degraded_rel_path = row["degraded_path"]
        degraded_abs_path = resolve_project_path(degraded_rel_path)

        try:
            image = load_image(degraded_abs_path)
        except Exception as e:
            print(f"[WARN] Не удалось открыть {degraded_abs_path}: {e}")
            preview_paths.append("")
            continue

        preview = build_preview(image, PREVIEW_SIZE)

        preview_filename = f"{sample_id}_preview.jpg"
        preview_output_path = PREVIEWS_DIR / preview_filename
        preview.save(preview_output_path, quality=95)

        preview_paths.append(relative_to_project(preview_output_path))
        saved_count += 1

    df["preview_path"] = preview_paths
    df.to_csv(csv_path, index=False)

    print(f"[OK] Preview generation finished.")
    print(f"[OK] Saved previews: {saved_count}")
    print(f"[OK] Updated CSV: {csv_path}")
    print(f"[OK] Preview dir: {PREVIEWS_DIR}")


if __name__ == "__main__":
    generate_previews()