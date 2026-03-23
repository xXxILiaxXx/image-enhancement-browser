from pathlib import Path


# Корень проекта
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Директории данных
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
DEGRADED_DIR = DATA_DIR / "degraded"
METADATA_DIR = DATA_DIR / "metadata"
SPLITS_DIR = DATA_DIR / "splits"
PREVIEWS_DIR = DATA_DIR / "previews"
PROCESSED_DIR = DATA_DIR / "processed"

# Артефакты
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"
ONNX_DIR = ARTIFACTS_DIR / "onnx"
LOGS_DIR = ARTIFACTS_DIR / "logs"
REPORTS_DIR = ARTIFACTS_DIR / "reports"

# Поддерживаемые расширения
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# Параметры генерации датасета
MAX_IMAGES = 5000
VERSIONS_PER_IMAGE = 2     # сколько плохих копий делать на 1 изображение
RANDOM_SEED = 42
PREVIEW_SIZE = (128, 128)

# Диапазоны деградации
BRIGHTNESS_RANGE = (-0.35, 0.35)   # additive, в долях от 255
CONTRAST_RANGE = (0.65, 1.35)      # multiplicative
SATURATION_RANGE = (0.65, 1.35)    # multiplicative