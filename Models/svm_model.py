from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor

from config import *
from common.data_utils import load_wind_series, prepare_supervised_data
from common.metrics import regression_metrics, save_results
from common.plot_utils import plot_prediction


def main() -> None:
    values, timestamps = load_wind_series(
        DATA_PATH, TARGET_COLUMN, DATE_COLUMN, HOUR_COLUMN, SHEET_NAME
    )
    data = prepare_supervised_data(values, timestamps, LOOKBACK, TRAIN_RATIO, VAL_RATIO)

    # 展平输入以适配 SVM
    X_train = data.x_train.reshape(data.x_train.shape[0], -1)
    X_val = data.x_val.reshape(data.x_val.shape[0], -1)
    X_test = data.x_test.reshape(data.x_test.shape[0], -1)

    # 使用 RBF 核，多输出回归器（针对 pred_len > 1 的情况）
    base_model = SVR(kernel="rbf", C=1.0, epsilon=0.1)
    model = MultiOutputRegressor(base_model)

    print("开始训练 SVM...")
    model.fit(X_train, data.y_train)

    pred_scaled = model.predict(X_test)
    if pred_scaled.ndim == 1:
        pred_scaled = pred_scaled.reshape(-1, 1)

    # 反归一化
    y_pred = data.scaler.inverse_transform(pred_scaled).reshape(-1)
    y_true = data.scaler.inverse_transform(data.y_test).reshape(-1)

    metrics = regression_metrics(y_true, y_pred)
    save_results("SVM", metrics, data.test_timestamps, y_true, y_pred, OUTPUT_DIR)

    result_df = pd.DataFrame({
        "time": data.test_timestamps,
        "actual": y_true,
        "prediction": y_pred
    })
    result_df.to_csv(
        OUTPUT_DIR / "SVM_prediction.csv",
        index=False,
        encoding="utf-8-sig"
    )

    plot_prediction("SVM", data.test_timestamps, y_true, y_pred, OUTPUT_DIR, PLOT_POINTS, SHOW_PLOTS)


if __name__ == "__main__":
    main()
