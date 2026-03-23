from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from app_ml.config.settings import METADATA_DIR, SPLITS_DIR, RANDOM_SEED


def save_split(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def make_splits() -> None:
    csv_path = METADATA_DIR / "dataset.csv"

    if not csv_path.exists():
        print(f"[ERROR] Не найден файл: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    if df.empty:
        print("[ERROR] dataset.csv пустой.")
        return

    # Сначала отделяем test
    train_val_df, test_df = train_test_split(
        df,
        test_size=0.15,
        random_state=RANDOM_SEED,
        shuffle=True,
    )

    # Потом из оставшегося отделяем val
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=0.1764705882,  # чтобы итогово получилось ~15% val от общего датасета
        random_state=RANDOM_SEED,
        shuffle=True,
    )

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    train_path = SPLITS_DIR / "train.csv"
    val_path = SPLITS_DIR / "val.csv"
    test_path = SPLITS_DIR / "test.csv"

    save_split(train_df, train_path)
    save_split(val_df, val_path)
    save_split(test_df, test_path)

    print("[OK] Split завершен.")
    print(f"[OK] train: {len(train_df)} -> {train_path}")
    print(f"[OK] val:   {len(val_df)} -> {val_path}")
    print(f"[OK] test:  {len(test_df)} -> {test_path}")
    print(f"[OK] total: {len(df)}")


if __name__ == "__main__":
    make_splits()