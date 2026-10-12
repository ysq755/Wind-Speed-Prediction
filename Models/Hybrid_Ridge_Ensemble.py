import os
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt


BASE_DIR = r"C:\Users\ysq75\Desktop\wind_speed_forecasting"
BP_FILE = os.path.join(BASE_DIR, "outputs", "BP_prediction.csv")
TF_FILE = os.path.join(BASE_DIR, "outputs", "Transformer_prediction.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# 用后 30% 作为验证集，前 70% 作为测试集
VAL_RATIO = 0.3

# Ridge 的候选 alpha（正则化强度），用内置交叉验证自动选最优
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]


def add_time_features(times):
    """从时间列提取小时特征，做成 sin/cos 编码（捕捉日周期）。"""
    t = pd.to_datetime(times)
    hour = t.hour.values  # 修改点：DatetimeIndex 没有 .dt，直接用 .hour
    sin_h = np.sin(2 * np.pi * hour / 24)
    cos_h = np.cos(2 * np.pi * hour / 24)
    return np.column_stack([sin_h, cos_h])


def main():
    bp = pd.read_csv(BP_FILE)
    tf = pd.read_csv(TF_FILE)

    # 长度对齐
    n = min(len(bp), len(tf))
    bp = bp.iloc[:n].reset_index(drop=True)
    tf = tf.iloc[:n].reset_index(drop=True)

    y_true = bp["actual"].values
    bp_pred = bp["prediction"].values
    tf_pred = tf["prediction"].values

    # ---- 构造特征矩阵 ----
    # 特征1：BP预测值
    # 特征2：Transformer预测值
    # 特征3/4：小时的 sin/cos 编码（让模型能区分白天/夜晚）
    time_feat = add_time_features(bp["time"].values)

    X = np.column_stack([bp_pred, tf_pred, time_feat])
    y = y_true

    print(f"\n特征矩阵形状: X={X.shape}, y={y.shape}")
    print(f"特征含义: [BP_pred, TF_pred, sin_hour, cos_hour]")

    # ---- 按时间顺序切分：后30%验证，前70%测试 ----
    split = int(n * (1 - VAL_RATIO))

    X_val, y_val = X[split:], y[split:]
    X_test, y_test = X[:split], y[:split]

    # ---- 用验证集训练 Ridge，自动搜 alpha ----
    # 注意：这里我们只用验证集来选超参，然后评估在测试集上
    model = RidgeCV(alphas=ALPHAS, cv=5)
    model.fit(X_val, y_val)

    print(f"\nRidge 最优 alpha: {model.alpha_:.4f}")
    print(f"系数: {dict(zip(['BP','TF','sin_h','cos_h'], model.coef_.round(4)))}")
    print(f"截距: {model.intercept_:.4f}")

    # ---- 在测试集上评估 ----
    pred_test = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, pred_test))
    mae = mean_absolute_error(y_test, pred_test)
    eps = 1e-8
    mape = np.mean(np.abs((y_test - pred_test) / (np.abs(y_test) + eps))) * 100
    r2 = r2_score(y_test, pred_test)

    print("\n====================")
    print("Hybrid Ridge Ensemble (BP + Transformer)")
    print("====================")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE : {mae:.6f}")
    print(f"MAPE: {mape:.4f}%")
    print(f"R2  : {r2:.6f}")

    # ---- 顺便把简单的等权平均和最优加权也跑出来做对比 ----
    # 等权平均
    pred_eq = 0.5 * bp_pred[:split] + 0.5 * tf_pred[:split]
    r2_eq = r2_score(y_test, pred_eq)

    # 最优单权重加权
    best_w, best_mse = 0.5, float("inf")
    for w in np.linspace(0, 1, 101):
        p = w * bp_pred[split:] + (1 - w) * tf_pred[split:]
        m = mean_squared_error(y_val, p)
        if m < best_mse:
            best_mse = m
            best_w = w
    pred_w = best_w * bp_pred[:split] + (1 - best_w) * tf_pred[:split]
    r2_w = r2_score(y_test, pred_w)

    print("\n---- 与其他融合方式对比 ----")
    print(f"等权平均 (0.5/0.5):        R2 = {r2_eq:.4f}")
    print(f"最优单权重 (w={best_w:.2f}):    R2 = {r2_w:.4f}")
    print(f"Ridge 融合:                R2 = {r2:.4f}")

    # ---- 保存 ----
    result = pd.DataFrame({
        "Actual": y_test,
        "Ridge_Ensemble": pred_test,
        "BP": bp_pred[:split],
        "Transformer": tf_pred[:split],
        "Equal_Weight": pred_eq,
        "Best_Weight": pred_w,
    })
    result.to_csv(
        os.path.join(OUTPUT_DIR, "Hybrid_Ridge_Ensemble_predictions.csv"),
        index=False, encoding="utf-8-sig"
    )

    metrics = pd.DataFrame({
        "Model": ["Hybrid Ridge Ensemble"],
        "Alpha": [model.alpha_],
        "RMSE": [rmse],
        "MAE": [mae],
        "MAPE": [mape],
        "R2": [r2]
    })
    metrics.to_csv(
        os.path.join(OUTPUT_DIR, "Hybrid_Ridge_Ensemble_metrics.csv"),
        index=False, encoding="utf-8-sig"
    )

    # ---- 绘图 ----
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(y_test, label="Actual", linewidth=1.5)
    plt.plot(pred_test, label="Ridge Ensemble", linewidth=1.5)
    plt.plot(bp_pred[:split], label="BP", alpha=0.4)
    plt.plot(tf_pred[:split], label="Transformer", alpha=0.4)
    plt.xlabel("Sample")
    plt.ylabel("Wind Speed")
    plt.title(f"Ridge Ensemble (R2={r2:.4f})")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    models = ["等权", f"最优单权({best_w:.2f})", "Ridge"]
    r2s = [r2_eq, r2_w, r2]
    colors = ["#999999", "#5B9BD5", "#ED7D31"]
    bars = plt.bar(models, r2s, color=colors)
    for bar, val in zip(bars, r2s):
        plt.text(bar.get_x() + bar.get_width()/2, val + 0.005,
                 f"{val:.4f}", ha="center", fontsize=10)
    plt.ylabel("R2")
    plt.title("融合方式对比")
    plt.ylim(0, max(r2s) * 1.2)
    plt.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "Hybrid_Ridge_Ensemble.png"),
        dpi=300
    )

    print("\n全部完成")


if __name__ == "__main__":
    main()
