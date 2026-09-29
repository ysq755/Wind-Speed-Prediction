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
# ============================================================
ROOT = Path(__file__).resolve().parent

RAW_DATA_PATH = (
    ROOT
    / "dataset"
    / "ETT-small"
    / "ETTh1.csv"
)

DENOISED_DATA_PATH = (
    ROOT
    / "dataset"
    / "ETT-small"
    / "ETTh1_denoised.csv"
)

RESULT_DIR = ROOT / "results"

CHECKPOINT_DIR = (
    ROOT
    / "checkpoints"
    / "meeting_ceemdan"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 1. Configuration
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

TRAIN_END = 12096
VAL_END = 14976

NUM_WORKERS = 0
PIN_MEMORY = torch.cuda.is_available()

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
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


def load_compare_data():
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find raw data:\n"
            f"{RAW_DATA_PATH}"
        )

    if not DENOISED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find denoised data:\n"
            f"{DENOISED_DATA_PATH}"
            f"\nRun run_ceemdan.py first."
        )

    raw_df = pd.read_csv(
        RAW_DATA_PATH
    )

    den_df = pd.read_csv(
        DENOISED_DATA_PATH
    )

    if len(raw_df) != len(den_df):
        raise ValueError(
            "Raw and denoised lengths differ."
        )

    raw_features = [
        col for col in raw_df.columns
        if col != "date"
    ]

    den_features = [
        col for col in den_df.columns
        if col != "date"
    ]

    if raw_features != den_features:
        raise ValueError(
            "Feature columns differ between "
            "raw and denoised files."
        )

    if TARGET_COL not in raw_features:
        raise ValueError(
            f"Cannot find target column "
            f"{TARGET_COL}"
        )

    target_idx = raw_features.index(
        TARGET_COL
    )

    raw_data = raw_df[
        raw_features
    ].values.astype(
        np.float64
    )

    den_data = den_df[
        den_features
    ].values.astype(
        np.float64
    )

    # Input scaler:
    # fit on denoised training inputs only.
    x_scaler = StandardScaler()

    x_scaler.fit(
        den_data[:TRAIN_END]
    )

    den_scaled = x_scaler.transform(
        den_data
    )

    # Target scaler:
    # fit on RAW OT training target only.
    y_scaler = StandardScaler()

    raw_train_target = raw_data[
        :TRAIN_END,
        target_idx
    ].reshape(-1, 1)

    y_scaler.fit(
        raw_train_target
    )

    raw_target_scaled = (
        y_scaler
        .transform(
            raw_data[
                :,
                target_idx
            ].reshape(-1, 1)
        )
        .reshape(-1)
    )

    return (
        raw_features,
        target_idx,
        den_scaled,
        raw_target_scaled,
        y_scaler
    )


def create_windows(
    x_data,
    y_target,
    start,
    end
):
    """
    Input:
        CEEMDAN-denoised historical sequence

    Target:
        RAW future OT

    This makes the comparison with raw baseline fair.
    """
    x_sub = (
        x_data[start:]
        if end is None
        else x_data[start:end]
    )

    y_sub = (
        y_target[start:]
        if end is None
        else y_target[start:end]
    )

    n_samples = (
        len(x_sub)
        - SEQ_LEN
        - PRED_LEN
        + 1
    )

    if n_samples <= 0:
        raise ValueError(
            "Split is too short."
        )

    X = np.empty(
        (
            n_samples,
            SEQ_LEN,
            x_sub.shape[1]
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

    for i in range(
        n_samples
    ):
        X[i] = x_sub[
            i:i + SEQ_LEN
        ]

        y[i] = y_sub[
            i + SEQ_LEN:
            i + SEQ_LEN + PRED_LEN
        ]

    return X, y


# ============================================================
# 4. Metrics
# ============================================================
def inverse_y(
    arr_scaled,
    y_scaler
):
    N, T = arr_scaled.shape

    restored = (
        y_scaler
        .inverse_transform(
            arr_scaled.reshape(-1, 1)
        )
        .reshape(N, T)
    )

    return restored


def calc_metrics(
    y_true,
    y_pred
):
    mse = np.mean(
        (y_true - y_pred) ** 2
    )

    rmse = np.sqrt(mse)

    mae = np.mean(
        np.abs(y_true - y_pred)
    )

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
# 5. Train / evaluate
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

        train_sum = 0.0

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

            train_sum += (
                loss.item()
            )

        train_loss = (
            train_sum
            / len(train_loader)
        )

        model.eval()

        val_sum = 0.0

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

                val_sum += (
                    criterion(
                        pred,
                        by
                    ).item()
                )

        val_loss = (
            val_sum
            / len(val_loader)
        )

        print(
            f"{model_name:<18} "
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
    y_scaler
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

    pred_scaled = np.concatenate(
        preds,
        axis=0
    )

    true_scaled = np.concatenate(
        labels,
        axis=0
    )

    pred_ori = inverse_y(
        pred_scaled,
        y_scaler
    )

    true_ori = inverse_y(
        true_scaled,
        y_scaler
    )

    return calc_metrics(
        true_ori,
        pred_ori
    )


# ============================================================
# 6. Main
# ============================================================
def main():
    set_seed()

    print("=" * 100)
    print(
        "STEP 3: CEEMDAN INPUT -> RAW FUTURE"
    )
    print("=" * 100)

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Raw: {RAW_DATA_PATH}"
    )

    print(
        f"Denoised: {DENOISED_DATA_PATH}"
    )

    (
        feature_cols,
        target_idx,
        den_scaled,
        raw_target_scaled,
        y_scaler
    ) = load_compare_data()

    X_train, y_train = create_windows(
        den_scaled,
        raw_target_scaled,
        0,
        TRAIN_END
    )

    X_val, y_val = create_windows(
        den_scaled,
        raw_target_scaled,
        TRAIN_END,
        VAL_END
    )

    X_test, y_test = create_windows(
        den_scaled,
        raw_target_scaled,
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
    print(
        "\nTraining CEEMDAN-LSTM..."
    )

    lstm = LSTMModel(
        config
    ).to(DEVICE)

    lstm = train_model(
        lstm,
        train_loader,
        val_loader,
        LR_LSTM,
        "ceemdan_lstm"
    )

    lstm_metrics = evaluate(
        lstm,
        test_loader,
        y_scaler
    )

    results.append(
        [
            "LSTM_CEEMDAN",
            *lstm_metrics
        ]
    )

    # ---------------- Transformer ----------------
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(
        "\nTraining CEEMDAN-Transformer..."
    )

    transformer = TransformerModel(
        config
    ).to(DEVICE)

    transformer = train_model(
        transformer,
        train_loader,
        val_loader,
        LR_TRANSFORMER,
        "ceemdan_transformer"
    )

    transformer_metrics = evaluate(
        transformer,
        test_loader,
        y_scaler
    )

    results.append(
        [
            "Transformer_CEEMDAN",
            *transformer_metrics
        ]
    )

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
    print(
        result_df.to_string(
            index=False
        )
    )
    print("=" * 100)

    ceemdan_result_path = (
        RESULT_DIR
        / "ceemdan_forecast_results.csv"
    )

    result_df.to_csv(
        ceemdan_result_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"\nSaved CEEMDAN forecast results to:\n"
        f"{ceemdan_result_path}"
    )

    # --------------------------------------------------------
    # Merge with baseline automatically
    # --------------------------------------------------------
    baseline_path = (
        RESULT_DIR
        / "raw_baseline_results.csv"
    )

    if baseline_path.exists():
        baseline_df = pd.read_csv(
            baseline_path
        )

        summary_df = pd.concat(
            [
                baseline_df,
                result_df
            ],
            ignore_index=True
        )

        summary_path = (
            RESULT_DIR
            / "final_comparison_summary.csv"
        )

        summary_df.to_csv(
            summary_path,
            index=False,
            encoding="utf-8-sig"
        )

        print(
            "\nFINAL SUMMARY"
        )

        print(
            summary_df.to_string(
                index=False
            )
        )

        print(
            f"\nSaved final summary to:\n"
            f"{summary_path}"
        )

    else:
        print(
            "\nWarning: baseline result not found."
            "\nRun run_baselines.py first."
        )


if __name__ == "__main__":
    main()
