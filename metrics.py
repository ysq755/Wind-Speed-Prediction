"""回归评价指标。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    # 风速可能为 0；使用极小值保护分母，避免除零错误。
    epsilon = 1e-6
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), epsilon))) * 100
    r2 = r2_score(y_true, y_pred)
    return {"MSE": float(mse), "MAE": float(mae), "MAPE(%)": float(mape), "R2": float(r2)}


def save_results(
    model_name: str,
    metrics: dict[str, float],
    timestamps,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n===== {model_name} 测试集指标 =====")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")

    with open(output_dir / f"{model_name}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    pd.DataFrame({
        "Timestamp": timestamps,
        "Actual Wind Speed (mph)": np.asarray(y_true).reshape(-1),
        "Predicted Wind Speed (mph)": np.asarray(y_pred).reshape(-1),
    }).to_csv(output_dir / f"{model_name}_predictions.csv", index=False, encoding="utf-8-sig")
