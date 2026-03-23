from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from app_ml.config.settings import CHECKPOINTS_DIR, PREVIEW_SIZE
from app_ml.models.cnn_regressor import CNNRegressor
from app_ml.utils.image_ops import load_image, save_image, degrade_image


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_preview(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    return image.resize(size, Image.Resampling.BILINEAR)


def load_model(model_path: Path, device: torch.device) -> CNNRegressor:
    model = CNNRegressor().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


@torch.no_grad()
def predict_correction(
    model: CNNRegressor,
    image: Image.Image,
    device: torch.device,
) -> tuple[float, float, float]:
    preview = build_preview(image, PREVIEW_SIZE)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    x = transform(preview).unsqueeze(0).to(device)
    pred = model(x).squeeze(0).cpu().numpy()

    brightness = float(pred[0])
    contrast = float(pred[1])
    saturation = float(pred[2])

    return brightness, contrast, saturation


def enhance_photo(
    input_path: Path,
    output_dir: Path,
    model_path: Path | None = None,
) -> Path:
    if not input_path.exists():
        raise FileNotFoundError(f"Не найдено изображение: {input_path}")

    if model_path is None:
        model_path = CHECKPOINTS_DIR / "cnn_regressor_best.pt"

    if not model_path.exists():
        raise FileNotFoundError(f"Не найдена модель: {model_path}")

    device = get_device()
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Model: {model_path}")

    model = load_model(model_path, device)

    original = load_image(input_path)

    pred_brightness, pred_contrast, pred_saturation = predict_correction(
        model=model,
        image=original,
        device=device,
    )

    corrected = degrade_image(
        image=original,
        brightness_delta=pred_brightness,
        contrast_factor=pred_contrast,
        saturation_factor=pred_saturation,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{input_path.stem}_enhanced.jpg"
    save_image(corrected, output_path)

    print("[OK] Prediction:")
    print(f"     brightness={pred_brightness:.4f}")
    print(f"     contrast={pred_contrast:.4f}")
    print(f"     saturation={pred_saturation:.4f}")
    print(f"[OK] Saved: {output_path}")

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enhance a local photo using the trained CNN model."
    )
    parser.add_argument(
        "image",
        type=str,
        help="Путь к изображению внутри проекта или абсолютный путь",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/reports/local_inference",
        help="Куда сохранить улучшенное изображение",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Путь к .pt модели. Если не указан, берется artifacts/checkpoints/cnn_regressor_best.pt",
    )

    args = parser.parse_args()

    input_path = Path(args.image).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve() if args.model else None

    enhance_photo(
        input_path=input_path,
        output_dir=output_dir,
        model_path=model_path,
    )


if __name__ == "__main__":
    main()