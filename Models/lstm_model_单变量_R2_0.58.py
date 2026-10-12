'''2026年8月份暑假版本'''

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch
from torch import nn

from config import *
from common.data_utils import load_wind_series, prepare_supervised_data
from common.metrics import regression_metrics, save_results
from common.plot_utils import plot_prediction
from common.torch_utils import predict_model, set_seed, train_model


class LSTMRegressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=LSTM_HIDDEN_SIZE,
            num_layers=LSTM_NUM_LAYERS,
            batch_first=True,
            dropout=DROPOUT if LSTM_NUM_LAYERS > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(LSTM_HIDDEN_SIZE),
            nn.Linear(LSTM_HIDDEN_SIZE, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        return self.head(output[:, -1, :])


def main() -> None:
    set_seed(SEED)
    values, timestamps = load_wind_series(
        DATA_PATH, TARGET_COLUMN, DATE_COLUMN, HOUR_COLUMN, SHEET_NAME
    )
    data = prepare_supervised_data(values, timestamps, LOOKBACK, TRAIN_RATIO, VAL_RATIO)

    model = LSTMRegressor()
    model, history = train_model(
        model,
        data.x_train,
        data.y_train,
        data.x_val,
        data.y_val,
        BATCH_SIZE,
        EPOCHS,
        PATIENCE,
        LEARNING_RATE,
        WEIGHT_DECAY,
        OUTPUT_DIR / "LSTM_best.pt",
    )
    pd.DataFrame(history).to_csv(OUTPUT_DIR / "LSTM_training_history.csv", index=False)

    pred_scaled = predict_model(model, data.x_test, BATCH_SIZE)
    y_pred = data.scaler.inverse_transform(pred_scaled).reshape(-1)
    y_true = data.scaler.inverse_transform(data.y_test).reshape(-1)

    metrics = regression_metrics(y_true, y_pred)
    save_results("LSTM", metrics, data.test_timestamps, y_true, y_pred, OUTPUT_DIR)
    plot_prediction("LSTM", data.test_timestamps, y_true, y_pred, OUTPUT_DIR, PLOT_POINTS, SHOW_PLOTS)


    # 保存LSTM预测结果
    result_df = pd.DataFrame({
        "time": data.test_timestamps,
        "actual": y_true,
        "prediction": y_pred
    }) #ChatGPT添加

    save_results(
        "LSTM",
        metrics,
        data.test_timestamps,
        y_true,
        y_pred,
        OUTPUT_DIR
    )

    # 保存预测结果
    result_df = pd.DataFrame({
        "time": data.test_timestamps,
        "actual": y_true,
        "prediction": y_pred
    })

    result_df.to_csv(
        OUTPUT_DIR / "LSTM_prediction.csv",
        index=False,
        encoding="utf-8-sig"
    )

    plot_prediction(
        "LSTM",
        data.test_timestamps,
        y_true,
        y_pred,
        OUTPUT_DIR,
        PLOT_POINTS,
        SHOW_PLOTS
    )

if __name__ == "__main__":
    main()

