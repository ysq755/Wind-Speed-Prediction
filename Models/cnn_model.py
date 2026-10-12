# # from __future__ import annotations
# #
# # from pathlib import Path
# # import sys
# #
# # PROJECT_ROOT = Path(__file__).resolve().parents[1]
# # if str(PROJECT_ROOT) not in sys.path:
# #     sys.path.insert(0, str(PROJECT_ROOT))
# #
# # import pandas as pd
# # import torch
# # from torch import nn
# #
# # from config import *
# # from common.data_utils import load_wind_series, prepare_supervised_data
# # from common.metrics import regression_metrics, save_results
# # from common.plot_utils import plot_prediction
# # from common.torch_utils import predict_model, set_seed, train_model
# #
# #
# # class CNN1DRegressor(nn.Module):
# #     """一维卷积神经网络，用于时间序列预测。"""
# #
# #     def __init__(self) -> None:
# #         super().__init__()
# #         self.conv_layers = nn.Sequential(
# #             # 输入: (batch, 1, LOOKBACK)
# #             nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
# #             nn.BatchNorm1d(32),
# #             nn.ReLU(),
# #             nn.MaxPool1d(kernel_size=2),
# #
# #             nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
# #             nn.BatchNorm1d(64),
# #             nn.ReLU(),
# #             nn.MaxPool1d(kernel_size=2),
# #
# #             nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
# #             nn.BatchNorm1d(128),
# #             nn.ReLU(),
# #             nn.AdaptiveAvgPool1d(output_size=1)
# #         )
# #         self.head = nn.Sequential(
# #             nn.Flatten(),
# #             nn.Linear(128, 64),
# #             nn.ReLU(),
# #             nn.Dropout(DROPOUT),
# #             nn.Linear(64, 1)
# #         )
# #
# #     def forward(self, x: torch.Tensor) -> torch.Tensor:
# #         # x 形状: (batch, LOOKBACK) -> (batch, 1, LOOKBACK)
# #         if x.dim() == 2:
# #             # x = x.unsqueeze(1)
# #             x = x.permute(0, 2, 1)   # <--- 加上这一行，交换后两维
# #         x = self.conv_layers(x)
# #         return self.head(x)
# #
# #
# # def main() -> None:
# #     set_seed(SEED)
# #     values, timestamps = load_wind_series(
# #         DATA_PATH, TARGET_COLUMN, DATE_COLUMN, HOUR_COLUMN, SHEET_NAME
# #     )
# #     data = prepare_supervised_data(values, timestamps, LOOKBACK, TRAIN_RATIO, VAL_RATIO)
# #
# #     model = CNN1DRegressor()
# #     model, history = train_model(
# #         model,
# #         data.x_train,
# #         data.y_train,
# #         data.x_val,
# #         data.y_val,
# #         BATCH_SIZE,
# #         EPOCHS,
# #         PATIENCE,
# #         LEARNING_RATE,
# #         WEIGHT_DECAY,
# #         OUTPUT_DIR / "CNN_best.pt",
# #     )
# #     pd.DataFrame(history).to_csv(OUTPUT_DIR / "CNN_training_history.csv", index=False)
# #
# #     pred_scaled = predict_model(model, data.x_test, BATCH_SIZE)
# #     y_pred = data.scaler.inverse_transform(pred_scaled).reshape(-1)
# #     y_true = data.scaler.inverse_transform(data.y_test).reshape(-1)
# #
# #     metrics = regression_metrics(y_true, y_pred)
# #     save_results("CNN", metrics, data.test_timestamps, y_true, y_pred, OUTPUT_DIR)
# #
# #     result_df = pd.DataFrame({
# #         "time": data.test_timestamps,
# #         "actual": y_true,
# #         "prediction": y_pred
# #     })
# #     result_df.to_csv(
# #         OUTPUT_DIR / "CNN_prediction.csv",
# #         index=False,
# #         encoding="utf-8-sig"
# #     )
# #
# #     plot_prediction("CNN", data.test_timestamps, y_true, y_pred, OUTPUT_DIR, PLOT_POINTS, SHOW_PLOTS)
# #
# #
# # if __name__ == "__main__":
# #     main()
#
#
# from __future__ import annotations
#
# from pathlib import Path
# import sys
#
# PROJECT_ROOT = Path(__file__).resolve().parents[1]
# if str(PROJECT_ROOT) not in sys.path:
#     sys.path.insert(0, str(PROJECT_ROOT))
#
# import pandas as pd
# import torch
# from torch import nn
#
# from config import *
# from common.data_utils import load_wind_series, prepare_supervised_data
# from common.metrics import regression_metrics, save_results
# from common.plot_utils import plot_prediction
# from common.torch_utils import predict_model, set_seed, train_model
#
#
# class CNN1DRegressor(nn.Module):
#     """一维卷积神经网络，用于时间序列预测。"""
#
#     def __init__(self) -> None:
#         super().__init__()
#         self.conv_layers = nn.Sequential(
#             # 输入: (batch, 1, LOOKBACK)
#             nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
#             nn.BatchNorm1d(32),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2),
#
#             nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2),
#
#             nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
#             nn.BatchNorm1d(128),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool1d(output_size=1)
#         )
#         self.head = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Dropout(DROPOUT),
#             nn.Linear(64, 1)
#         )
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # 统一把输入调整成 Conv1d 要求的 (batch, channels, seq_len)
#         # 情况1: (batch, LOOKBACK)          -> 升维成 (batch, 1, LOOKBACK)
#         # 情况2: (batch, LOOKBACK, 1)       -> 交换后两维成 (batch, 1, LOOKBACK)
#         if x.dim() == 2:
#             x = x.unsqueeze(1)
#         elif x.dim() == 3:
#             x = x.permute(0, 2, 1)
#         x = self.conv_layers(x)
#         return self.head(x)
#
#
# def main() -> None:
#     set_seed(SEED)
#     values, timestamps = load_wind_series(
#         DATA_PATH, TARGET_COLUMN, DATE_COLUMN, HOUR_COLUMN, SHEET_NAME
#     )
#     data = prepare_supervised_data(values, timestamps, LOOKBACK, TRAIN_RATIO, VAL_RATIO)
#
#     model = CNN1DRegressor()
#     model, history = train_model(
#         model,
#         data.x_train,
#         data.y_train,
#         data.x_val,
#         data.y_val,
#         BATCH_SIZE,
#         EPOCHS,
#         PATIENCE,
#         LEARNING_RATE,
#         WEIGHT_DECAY,
#         OUTPUT_DIR / "CNN_best.pt",
#     )
#     pd.DataFrame(history).to_csv(OUTPUT_DIR / "CNN_training_history.csv", index=False)
#
#     pred_scaled = predict_model(model, data.x_test, BATCH_SIZE)
#     y_pred = data.scaler.inverse_transform(pred_scaled).reshape(-1)
#     y_true = data.scaler.inverse_transform(data.y_test).reshape(-1)
#
#     metrics = regression_metrics(y_true, y_pred)
#     save_results("CNN", metrics, data.test_timestamps, y_true, y_pred, OUTPUT_DIR)
#
#     result_df = pd.DataFrame({
#         "time": data.test_timestamps,
#         "actual": y_true,
#         "prediction": y_pred
#     })
#     result_df.to_csv(
#         OUTPUT_DIR / "CNN_prediction.csv",
#         index=False,
#         encoding="utf-8-sig"
#     )
#
#     plot_prediction("CNN", data.test_timestamps, y_true, y_pred, OUTPUT_DIR, PLOT_POINTS, SHOW_PLOTS)
#
#
# if __name__ == "__main__":
#     main()

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


class CNN1DRegressor(nn.Module):
    """轻量化一维卷积神经网络，用于时间序列预测。

    相比原始三层卷积版本，砍掉了一层卷积，通道数减半，
    并加大了 Dropout，专门针对小样本数据防过拟合。
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv_layers = nn.Sequential(
            # 输入: (batch, 1, LOOKBACK)
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(output_size=1)
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 统一把输入调整成 Conv1d 要求的 (batch, channels, seq_len)
        # 情况1: (batch, LOOKBACK)     -> 升维成 (batch, 1, LOOKBACK)
        # 情况2: (batch, LOOKBACK, 1)  -> 交换后两维成 (batch, 1, LOOKBACK)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() == 3:
            x = x.permute(0, 2, 1)
        x = self.conv_layers(x)
        return self.head(x)


def main() -> None:
    set_seed(SEED)
    values, timestamps = load_wind_series(
        DATA_PATH, TARGET_COLUMN, DATE_COLUMN, HOUR_COLUMN, SHEET_NAME
    )
    data = prepare_supervised_data(values, timestamps, LOOKBACK, TRAIN_RATIO, VAL_RATIO)

    model = CNN1DRegressor()
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
        OUTPUT_DIR / "CNN_best.pt",
    )
    pd.DataFrame(history).to_csv(OUTPUT_DIR / "CNN_training_history.csv", index=False)

    pred_scaled = predict_model(model, data.x_test, BATCH_SIZE)
    y_pred = data.scaler.inverse_transform(pred_scaled).reshape(-1)
    y_true = data.scaler.inverse_transform(data.y_test).reshape(-1)

    metrics = regression_metrics(y_true, y_pred)
    save_results("CNN", metrics, data.test_timestamps, y_true, y_pred, OUTPUT_DIR)

    result_df = pd.DataFrame({
        "time": data.test_timestamps,
        "actual": y_true,
        "prediction": y_pred
    })
    result_df.to_csv(
        OUTPUT_DIR / "CNN_prediction.csv",
        index=False,
        encoding="utf-8-sig"
    )

    plot_prediction("CNN", data.test_timestamps, y_true, y_pred, OUTPUT_DIR, PLOT_POINTS, SHOW_PLOTS)


if __name__ == "__main__":
    main()
