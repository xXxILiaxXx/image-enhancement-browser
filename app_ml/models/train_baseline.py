from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from app_ml.config.settings import SPLITS_DIR, ARTIFACTS_DIR


FEATURE_COLUMNS = [
    "raw_mean_brightness",
    "raw_contrast_std",
    "raw_mean_saturation",
]

TARGET_COLUMNS = [
    "target_brightness",
    "target_contrast",
    "target_saturation",
]


def load_split(name: str) -> pd.DataFrame:
    path = SPLITS_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Не найден split: {path}")
    return pd.read_csv(path)


def evaluate_split(model: RandomForestRegressor, df: pd.DataFrame, split_name: str) -> None:
    x = df[FEATURE_COLUMNS]
    y_true = df[TARGET_COLUMNS]

    y_pred = model.predict(x)

    mae_per_target = mean_absolute_error(y_true, y_pred, multioutput="raw_values")
    mae_avg = mean_absolute_error(y_true, y_pred)

    print(f"\n[{split_name}]")
    print(f"MAE target_brightness: {mae_per_target[0]:.6f}")
    print(f"MAE target_contrast:   {mae_per_target[1]:.6f}")
    print(f"MAE target_saturation: {mae_per_target[2]:.6f}")
    print(f"MAE avg:               {mae_avg:.6f}")


def train_baseline() -> None:
    train_df = load_split("train")
    val_df = load_split("val")
    test_df = load_split("test")

    x_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMNS]

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(x_train, y_train)

    print("[OK] Baseline model trained.")

    evaluate_split(model, train_df, "TRAIN")
    evaluate_split(model, val_df, "VAL")
    evaluate_split(model, test_df, "TEST")

    output_dir = ARTIFACTS_DIR / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "baseline_rf.joblib"
    joblib.dump(model, model_path)

    print(f"\n[OK] Model saved: {model_path}")


if __name__ == "__main__":
    train_baseline()