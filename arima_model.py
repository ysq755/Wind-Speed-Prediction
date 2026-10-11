from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from config import *
from common.data_utils import load_wind_series
from common.metrics import regression_metrics, save_results
from common.plot_utils import plot_prediction


def main() -> None:
    values, timestamps = load_wind_series(
        DATA_PATH, TARGET_COLUMN, DATE_COLUMN, HOUR_COLUMN, SHEET_NAME
    )
    n = len(values)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    # 只在训练段估计参数。随后采用滚动一步预测：预测后追加真实观测，但不重估参数。
    # 默认 (24,0,0) 确保自回归部分使用前 24 个数据预测下一个数据。
    train_series = values[:train_end].astype(float)
    print(f"ARIMA order={ARIMA_ORDER}，训练参数估计样本数={len(train_series)}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = ARIMA(
            train_series,
            order=ARIMA_ORDER,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit()

    test_predictions = []
    test_actuals = []
    test_times = []

    # 从验证区间开始顺序更新模型状态；仅保存测试区间的预测。
    rolling_result = result
    for idx in range(train_end, n):
        pred = float(np.asarray(rolling_result.forecast(steps=1)).reshape(-1)[0])
        actual = float(values[idx])

        if idx >= val_end:
            test_predictions.append(pred)
            test_actuals.append(actual)
            test_times.append(timestamps[idx])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rolling_result = rolling_result.append([actual], refit=False)

        if (idx - train_end + 1) % 200 == 0:
            print(f"ARIMA rolling progress: {idx - train_end + 1}/{n - train_end}")

    y_true = np.asarray(test_actuals)
    y_pred = np.asarray(test_predictions)
    metrics = regression_metrics(y_true, y_pred)
    save_results("ARIMA", metrics, test_times, y_true, y_pred, OUTPUT_DIR)

    # ChatGPT添加开始-------------------------------------------------------------------------------------------------------
    # 保存ARIMA预测结果
    result_df = pd.DataFrame({
        "time": test_times,
        "actual": y_true,
        "prediction": y_pred
    })

    result_df.to_csv(
        OUTPUT_DIR / "ARIMA_prediction.csv",
        index=False,
        encoding="utf-8-sig"
    )
    # ChatGPT添加结束------------------------------------------------------------------------------------------------------------------------------


    plot_prediction("ARIMA", test_times, y_true, y_pred, OUTPUT_DIR, PLOT_POINTS, SHOW_PLOTS)


if __name__ == "__main__":
    main()

