"""数据读取、时间排序、标准化及滑动窗口构造。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class PreparedData:
    values: np.ndarray
    timestamps: pd.DatetimeIndex
    scaler: StandardScaler
    train_end: int
    val_end: int
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    test_timestamps: pd.DatetimeIndex


def _parse_timestamp(df: pd.DataFrame, date_column: str, hour_column: str) -> pd.Series:
    """解析 CIMIS 风格 Date + Hour；将 2400 处理为次日 00:00。"""
    dates = pd.to_datetime(df[date_column], errors="coerce")
    hours_num = pd.to_numeric(df[hour_column], errors="coerce")

    is_2400 = hours_num.eq(2400)
    hours = (hours_num // 100).fillna(0).astype(int)
    minutes = (hours_num % 100).fillna(0).astype(int)
    hours = hours.mask(is_2400, 0)
    minutes = minutes.mask(is_2400, 0)

    timestamps = (
        dates
        + pd.to_timedelta(hours, unit="h")
        + pd.to_timedelta(minutes, unit="m")
        + pd.to_timedelta(is_2400.astype(int), unit="D")
    )
    return timestamps


def load_wind_series(
    data_path: Path,
    target_column: str,
    date_column: str,
    hour_column: str,
    sheet_name: int | str = 0,
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """读取 Excel，按时间排序，返回风速序列与时间戳。"""
    if not data_path.exists():
        raise FileNotFoundError(f"找不到数据文件：{data_path}")

    df = pd.read_excel(data_path, sheet_name=sheet_name, engine="openpyxl")
    required = [target_column, date_column, hour_column]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"数据缺少列：{missing}；实际列名：{list(df.columns)}")

    work = pd.DataFrame({
        "timestamp": _parse_timestamp(df, date_column, hour_column),
        "target": pd.to_numeric(df[target_column], errors="coerce"),
    })
    work = work.dropna(subset=["timestamp", "target"])
    work = work.sort_values("timestamp")
    work = work.drop_duplicates(subset="timestamp", keep="last")

    if len(work) < 100:
        raise ValueError(f"清洗后只有 {len(work)} 条有效数据，数据量过少。")

    # 仅提示时间间隔，不强制重采样，确保“24 个数据预测 1 个数据”。
    deltas = work["timestamp"].diff().dropna()
    if not deltas.empty:
        common_delta = deltas.mode().iloc[0]
        irregular = int((deltas != common_delta).sum())
        print(f"主要采样间隔：{common_delta}；非主要间隔数量：{irregular}")

    values = work["target"].to_numpy(dtype=np.float32)
    timestamps = pd.DatetimeIndex(work["timestamp"])
    print(f"有效样本数：{len(values)}")
    print(f"时间范围：{timestamps.min()} 至 {timestamps.max()}")
    return values, timestamps


def _make_windows(
    scaled_values: np.ndarray,
    timestamps: pd.DatetimeIndex,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    x, y, target_indices, target_times = [], [], [], []
    for target_idx in range(lookback, len(scaled_values)):
        x.append(scaled_values[target_idx - lookback:target_idx])
        y.append(scaled_values[target_idx])
        target_indices.append(target_idx)
        target_times.append(timestamps[target_idx])

    return (
        np.asarray(x, dtype=np.float32)[..., None],
        np.asarray(y, dtype=np.float32).reshape(-1, 1),
        np.asarray(target_indices, dtype=np.int64),
        pd.DatetimeIndex(target_times),
    )


def prepare_supervised_data(
    values: np.ndarray,
    timestamps: pd.DatetimeIndex,
    lookback: int,
    train_ratio: float,
    val_ratio: float,
) -> PreparedData:
    """按目标时刻划分训练/验证/测试集，并仅用训练集拟合标准化器。"""
    n = len(values)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    if train_end <= lookback or val_end <= train_end or val_end >= n:
        raise ValueError("数据划分比例或数据量不合理。")

    scaler = StandardScaler()
    scaler.fit(values[:train_end].reshape(-1, 1))
    scaled = scaler.transform(values.reshape(-1, 1)).reshape(-1).astype(np.float32)

    x, y, target_indices, target_times = _make_windows(scaled, timestamps, lookback)
    train_mask = target_indices < train_end
    val_mask = (target_indices >= train_end) & (target_indices < val_end)
    test_mask = target_indices >= val_end

    print(
        f"窗口数量：训练 {train_mask.sum()}，验证 {val_mask.sum()}，测试 {test_mask.sum()}；"
        f"输入长度={lookback}，预测长度=1"
    )

    return PreparedData(
        values=values,
        timestamps=timestamps,
        scaler=scaler,
        train_end=train_end,
        val_end=val_end,
        x_train=x[train_mask],
        y_train=y[train_mask],
        x_val=x[val_mask],
        y_val=y[val_mask],
        x_test=x[test_mask],
        y_test=y[test_mask],
        test_timestamps=target_times[test_mask],
    )
