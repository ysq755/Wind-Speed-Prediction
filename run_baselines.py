import os
import random
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from models.LSTM import Model as LSTMModel
from models.TransformerMeeting import Model as TransformerModel

warnings.filterwarnings("ignore")


# ============================================================
# 0. Project paths
# Run this file from:
# D:\PyCharm_TSL\Time-Series-Library-main
# ============================================================
ROOT = Path(__file__).resolve().parent

DATA_PATH = ROOT / "dataset" / "ETT-small" / "ETTh1.csv"
RESULT_DIR = ROOT / "results"
CHECKPOINT_DIR = ROOT / "checkpoints" / "meeting_baseline"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. Experiment configuration
# Dell G15 5520 / 16GB RAM / RTX 3060
# ============================================================
SEED = 2026

SEQ_LEN = 96
PRED_LEN = 96

BATCH_SIZE = 16
EPOCHS = 30
PATIENCE = 6

D_MODEL = 64
E_LAYERS = 2
N_HEADS = 4
DROPOUT = 0.2

LR_LSTM = 1e-4
LR_TRANSFORMER = 3e-5

TARGET_COL = "OT"

# ETTh1 official split boundaries
TRAIN_END = 12096
VAL_END = 14976

NUM_WORKERS = 0
PIN_MEMORY = torch.cuda.is_available()

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# 2. Reproducibility
# ============================================================
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# 3. Dataset
# ============================================================
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(
            X,
            dtype=torch.float32
        )
        self.y = torch.tensor(
            y,
            dtype=torch.float32
        ).unsqueeze(-1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_raw_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find dataset: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    feature_cols = [
        col for col in df.columns
        if col != "date"
    ]

    if TARGET_COL not in feature_cols:
        raise ValueError(
            f"Cannot find target column {TARGET_COL}"
        )

    target_idx = feature_cols.index(
        TARGET_COL
    )

    data = df[
        feature_cols
    ].values.astype(np.float64)

    scaler = StandardScaler()

    # Fit only on training data
    scaler.fit(
        data[:TRAIN_END]
    )

    data_scaled = scaler.transform(
        data
    )

    return (
        df,
        feature_cols,
        target_idx,
        scaler,
        data_scaled
    )


def create_windows(
    data_scaled,
    target_idx,
    start,
    end
):
    """
    Build windows inside one split only.

    Input:
        past 96 steps, all features
    Target:
        future 96 steps, OT only
    """
    sub = (
        data_scaled[start:]
        if end is None
        else data_scaled[start:end]
    )

    n_samples = (
        len(sub)
        - SEQ_LEN
        - PRED_LEN
        + 1
    )

    if n_samples <= 0:
        raise ValueError(
            "Split is too short for window construction."
        )

    X = np.empty(
        (
            n_samples,
            SEQ_LEN,
            sub.shape[1]
        ),
        dtype=np.float32
    )

    y = np.empty(
        (
            n_samples,
            PRED_LEN
        ),
        dtype=np.float32
    )

    for i in range(n_samples):
        X[i] = sub[
            i:i + SEQ_LEN
        ]

        y[i] = sub[
            i + SEQ_LEN:
            i + SEQ_LEN + PRED_LEN,
            target_idx
        ]

    return X, y


# ============================================================
# 4. Metrics
# ============================================================
def inverse_target(
    arr_scaled,
    scaler,
    target_idx
):
    """
    arr_scaled: [N, T]
    """
    N, T = arr_scaled.shape

    dummy = np.zeros(
        (
            N * T,
            scaler.n_features_in_
        ),
        dtype=np.float64
    )

    dummy[:, target_idx] = (
        arr_scaled.reshape(-1)
    )

    restored = scaler.inverse_transform(
        dummy
    )[:, target_idx]

    return restored.reshape(
        N,
        T
    )


def calc_metrics(
    y_true,
    y_pred
):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mse = np.mean(
        (y_true - y_pred) ** 2
    )

    rmse = np.sqrt(mse)

    mae = np.mean(
        np.abs(y_true - y_pred)
    )

    # Global R2
    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)

    ss_res = np.sum(
        (yt - yp) ** 2
    )

    ss_tot = np.sum(
        (yt - np.mean(yt)) ** 2
    )

    global_r2 = (
        np.nan
        if ss_tot < 1e-12
        else 1 - ss_res / ss_tot
    )

    # Mean window-wise R2
    sample_r2 = []

    for i in range(
        y_true.shape[0]
    ):
        yt_i = y_true[i]
        yp_i = y_pred[i]

        ss_res_i = np.sum(
            (yt_i - yp_i) ** 2
        )

        ss_tot_i = np.sum(
            (
                yt_i
                - np.mean(yt_i)
            ) ** 2
        )

        if ss_tot_i >= 1e-12:
            sample_r2.append(
                1 - ss_res_i / ss_tot_i
            )

    mean_sample_r2 = (
        np.mean(sample_r2)
        if sample_r2
        else np.nan
    )

    return (
        mse,
        rmse,
        mae,
        global_r2,
        mean_sample_r2
    )


# ============================================================
# 5. Training
# ============================================================
def train_model(
    model,
    train_loader,
    val_loader,
    lr,
    model_name
):
    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    checkpoint_path = (
        CHECKPOINT_DIR
        / f"best_{model_name}.pth"
    )

    best_val = float("inf")
    trigger = 0

    for epoch in range(
        1,
        EPOCHS + 1
    ):
        model.train()

        train_loss_sum = 0.0

        for bx, by in train_loader:
            bx = bx.to(
                DEVICE,
                non_blocking=True
            )

            by = by.to(
                DEVICE,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            pred = model(
                bx,
                None,
                None,
                None
            )

            loss = criterion(
                pred,
                by
            )

            loss.backward()

            optimizer.step()

            train_loss_sum += (
                loss.item()
            )

        train_loss = (
            train_loss_sum
            / len(train_loader)
        )

        model.eval()

        val_loss_sum = 0.0

        with torch.no_grad():
            for bx, by in val_loader:
                bx = bx.to(
                    DEVICE,
                    non_blocking=True
                )

                by = by.to(
                    DEVICE,
                    non_blocking=True
                )

                pred = model(
                    bx,
                    None,
                    None,
                    None
                )

                val_loss_sum += (
                    criterion(
                        pred,
                        by
                    ).item()
                )

        val_loss = (
            val_loss_sum
            / len(val_loader)
        )

        print(
            f"{model_name:<12} "
            f"Epoch [{epoch:02d}/{EPOCHS}] "
            f"Train={train_loss:.6f} "
            f"Val={val_loss:.6f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            trigger = 0

            torch.save(
                model.state_dict(),
                checkpoint_path
            )

        else:
            trigger += 1

            if trigger >= PATIENCE:
                print(
                    f"Early stop: {model_name}, "
                    f"best val={best_val:.6f}"
                )
                break

    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=DEVICE
        )
    )

    return model


def evaluate(
    model,
    test_loader,
    scaler,
    target_idx
):
    model.eval()

    preds = []
    labels = []

    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(
                DEVICE,
                non_blocking=True
            )

            pred = model(
                bx,
                None,
                None,
                None
            )

            pred = (
                pred
                .cpu()
                .numpy()
                .squeeze(-1)
            )

            true = (
                by
                .numpy()
                .squeeze(-1)
            )

            preds.append(pred)
            labels.append(true)

    preds_scaled = np.concatenate(
        preds,
        axis=0
    )

    labels_scaled = np.concatenate(
        labels,
        axis=0
    )

    preds_ori = inverse_target(
        preds_scaled,
        scaler,
        target_idx
    )

    labels_ori = inverse_target(
        labels_scaled,
        scaler,
        target_idx
    )

    return calc_metrics(
        labels_ori,
        preds_ori
    )


# ============================================================
# 6. Main
# ============================================================
def main():
    set_seed()

    print("=" * 100)
    print("STEP 1: RAW BASELINE")
    print("=" * 100)
    print(f"Device: {DEVICE}")
    print(f"Dataset: {DATA_PATH}")
    print(
        f"seq_len={SEQ_LEN}, "
        f"pred_len={PRED_LEN}, "
        f"batch={BATCH_SIZE}, "
        f"d_model={D_MODEL}, "
        f"layers={E_LAYERS}"
    )

    (
        df,
        feature_cols,
        target_idx,
        scaler,
        data_scaled
    ) = load_raw_data()

    X_train, y_train = create_windows(
        data_scaled,
        target_idx,
        0,
        TRAIN_END
    )

    X_val, y_val = create_windows(
        data_scaled,
        target_idx,
        TRAIN_END,
        VAL_END
    )

    X_test, y_test = create_windows(
        data_scaled,
        target_idx,
        VAL_END,
        None
    )

    print(
        f"Samples: "
        f"train={len(X_train)}, "
        f"val={len(X_val)}, "
        f"test={len(X_test)}"
    )

    train_loader = DataLoader(
        TimeSeriesDataset(
            X_train,
            y_train
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    val_loader = DataLoader(
        TimeSeriesDataset(
            X_val,
            y_val
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    test_loader = DataLoader(
        TimeSeriesDataset(
            X_test,
            y_test
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    config = SimpleNamespace(
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN,
        enc_in=len(feature_cols),
        c_out=1,
        d_model=D_MODEL,
        e_layers=E_LAYERS,
        n_heads=N_HEADS,
        dropout=DROPOUT
    )

    results = []

    # ---------------- LSTM ----------------
    print("\nTraining Raw-LSTM...")

    lstm = LSTMModel(
        config
    ).to(DEVICE)

    lstm = train_model(
        lstm,
        train_loader,
        val_loader,
        LR_LSTM,
        "raw_lstm"
    )

    lstm_metrics = evaluate(
        lstm,
        test_loader,
        scaler,
        target_idx
    )

    results.append(
        [
            "LSTM_Raw",
            *lstm_metrics
        ]
    )

    # ---------------- Transformer ----------------
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(
        "\nTraining Raw-Transformer..."
    )

    transformer = TransformerModel(
        config
    ).to(DEVICE)

    transformer = train_model(
        transformer,
        train_loader,
        val_loader,
        LR_TRANSFORMER,
        "raw_transformer"
    )

    transformer_metrics = evaluate(
        transformer,
        test_loader,
        scaler,
        target_idx
    )

    results.append(
        [
            "Transformer_Raw",
            *transformer_metrics
        ]
    )

    # ---------------- Save ----------------
    result_df = pd.DataFrame(
        results,
        columns=[
            "Model",
            "MSE",
            "RMSE",
            "MAE",
            "Global_R2",
            "Mean_Sample_R2"
        ]
    )

    print("\n" + "=" * 100)
    print(result_df.to_string(index=False))
    print("=" * 100)

    save_path = (
        RESULT_DIR
        / "raw_baseline_results.csv"
    )

    result_df.to_csv(
        save_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"\nSaved baseline results to: "
        f"{save_path}"
    )


if __name__ == "__main__":
    main()
