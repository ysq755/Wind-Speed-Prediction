from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import math
import pandas as pd
import torch
from torch import nn

from config import *
from common.data_utils import load_wind_series, prepare_supervised_data
from common.metrics import regression_metrics, save_results
from common.plot_utils import plot_prediction
from common.torch_utils import predict_model, set_seed, train_model


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerRegressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_projection = nn.Linear(1, TRANSFORMER_D_MODEL)
        self.position = PositionalEncoding(TRANSFORMER_D_MODEL, LOOKBACK)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=TRANSFORMER_D_MODEL,
            nhead=TRANSFORMER_NHEAD,
            dim_feedforward=TRANSFORMER_DIM_FEEDFORWARD,
            dropout=DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=TRANSFORMER_NUM_LAYERS
        )
        self.head = nn.Sequential(
            nn.LayerNorm(TRANSFORMER_D_MODEL),
            nn.Linear(TRANSFORMER_D_MODEL, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        x = self.position(x)
        x = self.encoder(x)
        return self.head(x[:, -1, :])


def main() -> None:
    set_seed(SEED)
    values, timestamps = load_wind_series(
        DATA_PATH, TARGET_COLUMN, DATE_COLUMN, HOUR_COLUMN, SHEET_NAME
    )
    data = prepare_supervised_data(values, timestamps, LOOKBACK, TRAIN_RATIO, VAL_RATIO)

    model = TransformerRegressor()
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
        OUTPUT_DIR / "Transformer_best.pt",
    )
    pd.DataFrame(history).to_csv(
        OUTPUT_DIR / "Transformer_training_history.csv", index=False
    )

    pred_scaled = predict_model(model, data.x_test, BATCH_SIZE)
    y_pred = data.scaler.inverse_transform(pred_scaled).reshape(-1)
    y_true = data.scaler.inverse_transform(data.y_test).reshape(-1)

    metrics = regression_metrics(y_true, y_pred)
    save_results("Transformer", metrics, data.test_timestamps, y_true, y_pred, OUTPUT_DIR)

    # ChatGPT添加开始-------------------------------------------------------------------------------------------------------
    # 保存Transformer预测结果
    result_df = pd.DataFrame({
        "time": data.test_timestamps,
        "actual": y_true,
        "prediction": y_pred
    })

    result_df.to_csv(
        OUTPUT_DIR / "Transformer_prediction.csv",
        index=False,
        encoding="utf-8-sig"
    )
    # ChatGPT添加结束------------------------------------------------------------------------------------------------------------------------------

    plot_prediction(
        "Transformer", data.test_timestamps, y_true, y_pred,
        OUTPUT_DIR, PLOT_POINTS, SHOW_PLOTS
    )


if __name__ == "__main__":
    main()

