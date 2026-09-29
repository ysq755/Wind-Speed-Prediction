import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PyEMD import CEEMDAN

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

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 1. CEEMDAN configuration
#
# Meeting version:
#   - remove IMF1 only
#   - lower trials for faster CPU execution
#
# For final paper experiments, increase TRIALS later.
# ============================================================
TARGET_COL = "OT"

TRAIN_END = 12096
VAL_END = 14976

REMOVE_HIGH_FREQ_IMFS = 1

TRIALS = 20
EPSILON = 0.05

SEED = 2026

PLOT_START = 0
PLOT_LEN = 500


# ============================================================
# 2. CEEMDAN
# ============================================================
def ceemdan_denoise(
    series,
    remove_high_freq_imfs=1
):
    """
    CEEMDAN IMFs are generally ordered
    from high frequency to low frequency.

    Denoising:
        denoised
        = original
        - IMF1
        - IMF2
        - ...

    Do NOT reconstruct denoised signal by
    summing the first few IMFs.
    """
    series = np.asarray(
        series,
        dtype=np.float64
    )

    ceemdan = CEEMDAN(
        trials=TRIALS,
        epsilon=EPSILON
    )

    if hasattr(
        ceemdan,
        "noise_seed"
    ):
        ceemdan.noise_seed(
            SEED
        )

    imfs = ceemdan.ceemdan(
        series
    )

    if (
        imfs.ndim != 2
        or imfs.shape[1] != len(series)
    ):
        raise RuntimeError(
            f"Unexpected CEEMDAN shape: "
            f"{imfs.shape}"
        )

    n_imfs = imfs.shape[0]

    remove_n = min(
        remove_high_freq_imfs,
        max(
            n_imfs - 1,
            0
        )
    )

    if remove_n == 0:
        removed_noise = (
            np.zeros_like(series)
        )

        denoised = series.copy()

    else:
        removed_noise = np.sum(
            imfs[:remove_n],
            axis=0
        )

        denoised = (
            series
            - removed_noise
        )

    return (
        denoised,
        imfs,
        removed_noise
    )


def calc_stats(
    raw,
    denoised
):
    mse = np.mean(
        (raw - denoised) ** 2
    )

    mae = np.mean(
        np.abs(raw - denoised)
    )

    signal_power = np.mean(
        raw ** 2
    )

    noise_power = np.mean(
        (raw - denoised) ** 2
    )

    snr = (
        np.inf
        if noise_power < 1e-12
        else 10
        * np.log10(
            signal_power
            / noise_power
        )
    )

    corr = np.corrcoef(
        raw,
        denoised
    )[0, 1]

    return (
        mse,
        mae,
        snr,
        corr
    )


def process_split(
    split_name,
    series
):
    print(
        f"\nProcessing {split_name}: "
        f"length={len(series)}"
    )

    (
        denoised,
        imfs,
        removed_noise
    ) = ceemdan_denoise(
        series,
        REMOVE_HIGH_FREQ_IMFS
    )

    (
        mse,
        mae,
        snr,
        corr
    ) = calc_stats(
        series,
        denoised
    )

    print(
        f"Number of IMFs: "
        f"{imfs.shape[0]}"
    )

    print(
        f"Removed: first "
        f"{REMOVE_HIGH_FREQ_IMFS} IMF(s)"
    )

    print(
        f"Raw mean/std: "
        f"{series.mean():.4f} / "
        f"{series.std():.4f}"
    )

    print(
        f"Denoised mean/std: "
        f"{denoised.mean():.4f} / "
        f"{denoised.std():.4f}"
    )

    print(
        f"MSE(raw,den): "
        f"{mse:.6f}"
    )

    print(
        f"MAE(raw,den): "
        f"{mae:.6f}"
    )

    print(
        f"SNR(dB): "
        f"{snr:.4f}"
    )

    print(
        f"Correlation: "
        f"{corr:.6f}"
    )

    return denoised


# ============================================================
# 3. Main
# ============================================================
def main():
    print("=" * 100)
    print("STEP 2: CEEMDAN DENOISING")
    print("=" * 100)

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find: "
            f"{RAW_DATA_PATH}"
        )

    df = pd.read_csv(
        RAW_DATA_PATH
    )

    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Cannot find target column: "
            f"{TARGET_COL}"
        )

    if len(df) <= VAL_END:
        raise ValueError(
            "Dataset length is shorter "
            "than ETTh1 split boundaries."
        )

    raw_ot = df[
        TARGET_COL
    ].values.astype(
        np.float64
    )

    # --------------------------------------------------------
    # Train / validation / test are decomposed separately.
    #
    # This avoids direct cross-split decomposition.
    #
    # IMPORTANT:
    # This is a lightweight meeting experiment,
    # NOT strict causal rolling CEEMDAN.
    # Strict rolling decomposition will be much slower.
    # --------------------------------------------------------
    train_raw = raw_ot[
        :TRAIN_END
    ]

    val_raw = raw_ot[
        TRAIN_END:VAL_END
    ]

    test_raw = raw_ot[
        VAL_END:
    ]

    train_den = process_split(
        "TRAIN",
        train_raw
    )

    val_den = process_split(
        "VALIDATION",
        val_raw
    )

    test_den = process_split(
        "TEST",
        test_raw
    )

    denoised_all = np.concatenate(
        [
            train_den,
            val_den,
            test_den
        ]
    )

    if len(
        denoised_all
    ) != len(df):
        raise RuntimeError(
            "Denoised length mismatch."
        )

    output_df = df.copy()

    # Only OT is denoised.
    # Other variables stay unchanged.
    output_df[
        TARGET_COL
    ] = denoised_all

    output_df.to_csv(
        DENOISED_DATA_PATH,
        index=False
    )

    print(
        f"\nSaved denoised dataset to:\n"
        f"{DENOISED_DATA_PATH}"
    )

    # --------------------------------------------------------
    # Sanity-check plot
    # --------------------------------------------------------
    start = PLOT_START

    end = min(
        len(raw_ot),
        start + PLOT_LEN
    )

    plt.figure(
        figsize=(14, 6)
    )

    plt.plot(
        np.arange(
            start,
            end
        ),
        raw_ot[
            start:end
        ],
        label="Raw OT",
        linewidth=1.2
    )

    plt.plot(
        np.arange(
            start,
            end
        ),
        denoised_all[
            start:end
        ],
        label=(
            "CEEMDAN "
            f"(remove IMF1-"
            f"{REMOVE_HIGH_FREQ_IMFS})"
        ),
        linewidth=1.6
    )

    plt.xlabel(
        "Time step"
    )

    plt.ylabel(
        "OT"
    )

    plt.title(
        "Raw OT vs CEEMDAN-denoised OT"
    )

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    figure_path = (
        RESULT_DIR
        / "ceemdan_raw_vs_denoised.png"
    )

    plt.savefig(
        figure_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved sanity-check figure to:\n"
        f"{figure_path}"
    )

    print(
        "\nIMPORTANT:"
        "\nOpen this figure before running "
        "run_compare.py."
        "\nThe denoised line should preserve "
        "the Raw OT trend while removing "
        "small high-frequency fluctuations."
    )


if __name__ == "__main__":
    main()
