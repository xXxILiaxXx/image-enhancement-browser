from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error

from app_ml.config.settings import CHECKPOINTS_DIR
from app_ml.data.preview_dataset import PreviewRegressionDataset
from app_ml.models.cnn_regressor import CNNRegressor


BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 1e-3


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model: CNNRegressor, loader: DataLoader, device: torch.device, split_name: str) -> float:
    model.eval()

    all_targets = []
    all_preds = []

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        preds = model(images)

        all_targets.append(targets.cpu())
        all_preds.append(preds.cpu())

    y_true = torch.cat(all_targets).numpy()
    y_pred = torch.cat(all_preds).numpy()

    mae_per_target = mean_absolute_error(y_true, y_pred, multioutput="raw_values")
    mae_avg = mean_absolute_error(y_true, y_pred)

    print(f"\n[{split_name}]")
    print(f"MAE target_brightness: {mae_per_target[0]:.6f}")
    print(f"MAE target_contrast:   {mae_per_target[1]:.6f}")
    print(f"MAE target_saturation: {mae_per_target[2]:.6f}")
    print(f"MAE avg:               {mae_avg:.6f}")

    return mae_avg


def train() -> None:
    device = get_device()
    print(f"[INFO] Device: {device}")

    train_ds = PreviewRegressionDataset("train")
    val_ds = PreviewRegressionDataset("val")
    test_ds = PreviewRegressionDataset("test")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = CNNRegressor().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_mae = float("inf")

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    best_model_path = CHECKPOINTS_DIR / "cnn_regressor_best.pt"

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        print(f"\n[Epoch {epoch}/{EPOCHS}] train_loss={avg_train_loss:.6f}")

        val_mae = evaluate(model, val_loader, device, "VAL")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), best_model_path)
            print(f"[OK] Saved best model: {best_model_path}")

    print("\n[INFO] Loading best model for final test...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    evaluate(model, test_loader, device, "TEST")


if __name__ == "__main__":
    train()