from __future__ import annotations

import colorsys
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageStat


# ----------------------------
# I/O
# ----------------------------

def load_image(image_path: Path) -> Image.Image:
    """Загружает изображение и приводит к RGB."""
    return Image.open(image_path).convert("RGB")


def save_image(image: Image.Image, output_path: Path) -> None:
    """Сохраняет изображение, создавая папки при необходимости."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=95)


# ----------------------------
# Basic image transforms
# ----------------------------

def apply_brightness(image: Image.Image, brightness_delta: float) -> Image.Image:
    """
    brightness_delta:
        0.0  -> без изменений
        -0.2 -> затемнить
        +0.2 -> осветлить
    """
    enhancer = ImageEnhance.Brightness(image)
    factor = 1.0 + brightness_delta
    factor = max(0.1, factor)
    return enhancer.enhance(factor)


def apply_contrast(image: Image.Image, contrast_factor: float) -> Image.Image:
    """
    contrast_factor:
        1.0  -> без изменений
        <1.0 -> снизить контраст
        >1.0 -> повысить контраст
    """
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(max(0.1, contrast_factor))


def apply_saturation(image: Image.Image, saturation_factor: float) -> Image.Image:
    """
    saturation_factor:
        1.0  -> без изменений
        <1.0 -> уменьшить насыщенность
        >1.0 -> увеличить насыщенность
    """
    enhancer = ImageEnhance.Color(image)
    return enhancer.enhance(max(0.1, saturation_factor))


def degrade_image(
    image: Image.Image,
    brightness_delta: float,
    contrast_factor: float,
    saturation_factor: float,
) -> Image.Image:
    """Применяет деградации последовательно."""
    result = image.copy()
    result = apply_brightness(result, brightness_delta)
    result = apply_contrast(result, contrast_factor)
    result = apply_saturation(result, saturation_factor)
    return result


# ----------------------------
# Raw image statistics
# ----------------------------

def compute_mean_brightness(image: Image.Image) -> float:
    """
    Средняя яркость изображения в диапазоне [0, 1].
    """
    grayscale = image.convert("L")
    stat = ImageStat.Stat(grayscale)
    return stat.mean[0] / 255.0


def compute_contrast_std(image: Image.Image) -> float:
    """
    Контраст как стандартное отклонение яркости в диапазоне [0, 1].
    Чем меньше значение, тем картинка более плоская.
    """
    grayscale = image.convert("L")
    arr = np.asarray(grayscale, dtype=np.float32) / 255.0
    return float(arr.std())


def compute_mean_saturation(image: Image.Image) -> float:
    """
    Средняя насыщенность изображения в диапазоне [0, 1].
    Считается через преобразование RGB -> HSV.
    """
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    flat = rgb.reshape(-1, 3)

    saturations = []
    for r, g, b in flat:
        _, s, _ = colorsys.rgb_to_hsv(float(r), float(g), float(b))
        saturations.append(s)

    return float(np.mean(saturations))


# ----------------------------
# Image state classification
# ----------------------------

def classify_image_state(
    mean_brightness: float,
    contrast_std: float,
    mean_saturation: float,
) -> str:
    """
    Назначает исходному фото один главный state.

    Приоритет специально задан так, чтобы сильные отклонения
    определялись раньше более нейтральных состояний.
    """
    if mean_brightness < 0.28:
        return "dark"
    if mean_brightness > 0.52:
        return "bright"
    if contrast_std < 0.17:
        return "low_contrast"
    if mean_saturation < 0.20:
        return "desaturated"
    if contrast_std > 0.28:
        return "high_contrast"
    if mean_saturation > 0.55:
        return "high_saturation"
    return "balanced"


# ----------------------------
# Degradation profiles
# ----------------------------

def choose_degradation_profile(raw_state: str) -> str:
    """
    Выбирает профиль деградации в зависимости от состояния исходного фото.
    Мы стараемся НЕ улучшить изображение случайно, а увести его дальше
    от визуального баланса.
    """
    state_to_profiles = {
        "dark": [
            "too_dark",
            "low_contrast_dark",
            "desaturated_dark",
        ],
        "bright": [
            "overexposed",
            "washed_out",
            "low_contrast_bright",
        ],
        "low_contrast": [
            "washed_out",
            "desaturated",
        ],
        "desaturated": [
            "more_desaturated",
            "washed_out",
        ],
        "high_contrast": [
            "too_harsh",
            "oversaturated",
        ],
        "high_saturation": [
            "oversaturated",
            "too_harsh",
        ],
        "balanced": [
            "too_dark",
            "overexposed",
            "washed_out",
            "desaturated",
            "oversaturated",
            "too_harsh",
        ],
    }

    profiles = state_to_profiles.get(raw_state, ["washed_out"])
    return random.choice(profiles)


def sample_profile_params(profile: str) -> tuple[float, float, float]:
    """
    Возвращает параметры деградации для конкретного плохого сценария.

    Возвращаем:
        brightness_delta
        contrast_factor
        saturation_factor
    """
    if profile == "too_dark":
        brightness = random.uniform(-0.35, -0.18)
        contrast = random.uniform(0.90, 1.05)
        saturation = random.uniform(0.85, 1.00)

    elif profile == "low_contrast_dark":
        brightness = random.uniform(-0.25, -0.10)
        contrast = random.uniform(0.65, 0.85)
        saturation = random.uniform(0.80, 0.95)

    elif profile == "desaturated_dark":
        brightness = random.uniform(-0.22, -0.08)
        contrast = random.uniform(0.85, 1.00)
        saturation = random.uniform(0.45, 0.75)

    elif profile == "overexposed":
        brightness = random.uniform(0.18, 0.35)
        contrast = random.uniform(0.75, 0.95)
        saturation = random.uniform(0.80, 0.98)

    elif profile == "washed_out":
        brightness = random.uniform(0.08, 0.22)
        contrast = random.uniform(0.60, 0.82)
        saturation = random.uniform(0.55, 0.80)

    elif profile == "low_contrast_bright":
        brightness = random.uniform(0.10, 0.24)
        contrast = random.uniform(0.62, 0.85)
        saturation = random.uniform(0.75, 0.95)

    elif profile == "desaturated":
        brightness = random.uniform(-0.05, 0.08)
        contrast = random.uniform(0.85, 1.00)
        saturation = random.uniform(0.40, 0.70)

    elif profile == "more_desaturated":
        brightness = random.uniform(-0.03, 0.06)
        contrast = random.uniform(0.88, 1.02)
        saturation = random.uniform(0.25, 0.55)

    elif profile == "oversaturated":
        brightness = random.uniform(-0.05, 0.08)
        contrast = random.uniform(0.95, 1.12)
        saturation = random.uniform(1.25, 1.55)

    elif profile == "too_harsh":
        brightness = random.uniform(-0.05, 0.05)
        contrast = random.uniform(1.20, 1.45)
        saturation = random.uniform(1.00, 1.20)

    else:
        # безопасный fallback
        brightness = random.uniform(0.05, 0.15)
        contrast = random.uniform(0.70, 0.90)
        saturation = random.uniform(0.70, 0.90)

    return brightness, contrast, saturation


def sample_degradation_for_image(
    mean_brightness: float,
    contrast_std: float,
    mean_saturation: float,
) -> tuple[str, str, float, float, float]:
    """
    Полный пайплайн выбора деградации для конкретного изображения:
    1. определяем state исходника
    2. выбираем degradation profile
    3. сэмплируем параметры под этот profile
    """
    raw_state = classify_image_state(
        mean_brightness=mean_brightness,
        contrast_std=contrast_std,
        mean_saturation=mean_saturation,
    )
    degradation_profile = choose_degradation_profile(raw_state)
    brightness_delta, contrast_factor, saturation_factor = sample_profile_params(
        degradation_profile
    )

    return (
        raw_state,
        degradation_profile,
        brightness_delta,
        contrast_factor,
        saturation_factor,
    )


# ----------------------------
# Targets
# ----------------------------

def make_correction_targets(
    brightness_delta: float,
    contrast_factor: float,
    saturation_factor: float,
) -> tuple[float, float, float]:
    """
    Таргеты, которые потом должна предсказывать модель.
    """
    target_brightness = -brightness_delta
    target_contrast = 1.0 / contrast_factor
    target_saturation = 1.0 / saturation_factor
    return target_brightness, target_contrast, target_saturation


# ----------------------------
# Filesystem helpers
# ----------------------------

def is_image_file(path: Path, allowed_extensions: set[str]) -> bool:
    return path.is_file() and path.suffix.lower() in allowed_extensions


def list_images(directory: Path, allowed_extensions: set[str]) -> list[Path]:
    return sorted(
        [p for p in directory.rglob("*") if is_image_file(p, allowed_extensions)]
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)