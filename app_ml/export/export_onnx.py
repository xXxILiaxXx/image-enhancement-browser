from __future__ import annotations

import torch

from app_ml.config.settings import CHECKPOINTS_DIR, ONNX_DIR, PREVIEW_SIZE
from app_ml.models.cnn_regressor import CNNRegressor


def export_onnx() -> None:
    model_path = CHECKPOINTS_DIR / "cnn_regressor_best.pt"
    onnx_path = ONNX_DIR / "cnn_regressor.onnx"

    if not model_path.exists():
        print(f"[ERROR] Не найдена модель: {model_path}")
        return

    ONNX_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    print(f"[INFO] Export device: {device}")

    model = CNNRegressor().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    dummy_input = torch.randn(1, 3, PREVIEW_SIZE[1], PREVIEW_SIZE[0], device=device)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        external_data=False,   # <-- главное
    )

    print(f"[OK] ONNX model exported: {onnx_path}")


if __name__ == "__main__":
    export_onnx()