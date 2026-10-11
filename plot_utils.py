"""真实值与预测值折线对比图。"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_prediction(
    model_name: str,
    timestamps,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: Path,
    plot_points: int | None = 300,
    show: bool = True,
) -> None:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    if plot_points is not None and plot_points > 0:
        n = min(plot_points, len(y_true))
        timestamps = timestamps[:n]
        y_true = y_true[:n]
        y_pred = y_pred[:n]

    plt.figure(figsize=(14, 5.5))
    plt.plot(timestamps, y_true, label="Actual", linewidth=1.6)
    plt.plot(timestamps, y_pred, label="Predicted", linewidth=1.3)
    plt.title(f"{model_name}: Actual vs Predicted Wind Speed")
    plt.xlabel("Time")
    plt.ylabel("Wind Speed (mph)")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name}_actual_vs_predicted.png", dpi=300)

    # 模型脚本运行到此处会自动弹出图，不需要额外调用画图脚本。
    if show:
        plt.show()
    plt.close()
