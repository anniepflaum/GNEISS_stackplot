from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat

try:
    from .resampling import TIME_STEP_S, reduce_series_resolution
except ImportError:
    from resampling import TIME_STEP_S, reduce_series_resolution


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parents[1] / "data"
APP_DATA_DIR = DATA_DIR / "app_data"
ERAU_MAT = DATA_DIR / "source_data" / "ERAU_Raw_signal_export.mat"
ERAU_NPZ = APP_DATA_DIR / "erau_signal_data.npz"
TG_TO_MAGLAT_CSV = APP_DATA_DIR / "tg_to_maglat.csv"
SIGNAL_VARIABLE = "Signal_397_398"
TIME_VARIABLE = "Time_TG_TG398"
SERIES_INDICES = {
    "397D2": 0,
    "398D2": 2,
}
T0_TG_OFFSET_S = {
    "397D2": 0.0,
    "398D2": 30.0,
}
ROCKET_COLORS = {
    "397": "tab:blue",
    "398": "tab:orange",
}
MEDIAN_WINDOW_SAMPLES = 100


def load_erau_signals(mat_path: str | Path = ERAU_MAT):
    """Load the ERAU PIP D2 signal series and their TG times."""
    data = loadmat(
        Path(mat_path),
        variable_names=[SIGNAL_VARIABLE, TIME_VARIABLE],
        squeeze_me=True,
    )
    signals = data[SIGNAL_VARIABLE]
    times = data[TIME_VARIABLE]
    series_data = {}

    for label, series_index in SERIES_INDICES.items():
        time_since_tg_s = np.asarray(times[series_index], dtype=float).ravel()
        signal = np.asarray(signals[series_index], dtype=float).ravel()
        if time_since_tg_s.shape != signal.shape:
            raise ValueError(
                f"ERAU {label} time and signal shapes differ: "
                f"{time_since_tg_s.shape} != {signal.shape}"
            )
        series_data[label] = (time_since_tg_s, signal)

    return series_data


def sort_unique_time(time_since_tg_s, signal):
    """Sort a signal by TG time and keep the first sample at duplicate times."""
    order = np.argsort(time_since_tg_s, kind="stable")
    time_since_tg_s = time_since_tg_s[order]
    signal = signal[order]
    time_since_tg_s, unique_indices = np.unique(
        time_since_tg_s,
        return_index=True,
    )
    return time_since_tg_s, signal[unique_indices]


def rolling_median(signal, window_samples: int = MEDIAN_WINDOW_SAMPLES):
    """Return a centered rolling median using the nearest signal samples."""
    return (
        pd.Series(signal)
        .rolling(window=window_samples, center=True, min_periods=1)
        .median()
        .to_numpy()
    )


def processed_erau_signals(mat_path: str | Path = ERAU_MAT):
    """Load, sort, and median-filter the ERAU PIP D2 signals."""
    series_data = {}

    for label, (time_since_tg_s, signal) in load_erau_signals(mat_path).items():
        time_since_tg_s, signal = sort_unique_time(time_since_tg_s, signal)
        time_since_tg_s = time_since_tg_s + T0_TG_OFFSET_S[label]
        series_data[label] = (time_since_tg_s, rolling_median(signal))

    return series_data


def load_maglat_mapping(
    csv_path: str | Path = TG_TO_MAGLAT_CSV,
):
    """Load TG-to-magnetic-latitude mappings for both rockets."""
    data = np.genfromtxt(
        Path(csv_path),
        delimiter=",",
        names=True,
        dtype=float,
        encoding=None,
    )
    mappings = {}

    for rocket in ("397", "398"):
        maglat = data[f"{rocket}_magnetic_lat_deg"]
        valid = np.isfinite(maglat)
        mappings[rocket] = (data["time_since_TG_s"][valid], maglat[valid])

    return mappings


def interpolate_maglat(time_since_tg_s, rocket: str, mappings=None):
    """Interpolate one rocket's magnetic latitude onto a TG time array."""
    if mappings is None:
        mappings = load_maglat_mapping()

    mapping_time_s, mapping_maglat_deg = mappings[rocket]
    return np.interp(
        time_since_tg_s,
        mapping_time_s,
        mapping_maglat_deg,
        left=np.nan,
        right=np.nan,
    )


def export_erau_npz(
    output_path: str | Path = ERAU_NPZ,
    mat_path: str | Path = ERAU_MAT,
):
    """Export median-filtered ERAU signal arrays with source-file metadata."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    series = []
    maglat_mappings = load_maglat_mapping()

    for label, (time_since_tg_s, signal) in processed_erau_signals(mat_path).items():
        rocket = label[:-2]
        time_since_tg_s, signal = reduce_series_resolution(time_since_tg_s, signal)
        time_key = f"{label}_time_since_TG_s"
        maglat_key = f"{label}_magnetic_lat_deg"
        value_key = f"{label}_signal"
        arrays[time_key] = time_since_tg_s
        arrays[maglat_key] = interpolate_maglat(
            time_since_tg_s,
            rocket,
            mappings=maglat_mappings,
        )
        arrays[value_key] = signal
        series.append(
            {
                "label": label,
                "time_key": time_key,
                "maglat_key": maglat_key,
                "value_key": value_key,
                "units": "nA",
            }
        )

    metadata = {
        "source_data_file": Path(mat_path).name,
        "source_signal_variable": SIGNAL_VARIABLE,
        "source_time_variable": TIME_VARIABLE,
        "maglat_mapping_file": TG_TO_MAGLAT_CSV.name,
        "median_window_samples": MEDIAN_WINDOW_SAMPLES,
        "t0_tg_offset_s": T0_TG_OFFSET_S,
        "maximum_time_resolution_s": TIME_STEP_S,
        "series": series,
    }
    np.savez_compressed(
        output_path,
        **arrays,
        metadata_json=np.array(json.dumps(metadata)),
    )

    return output_path


def plot_erau(
    mat_path: str | Path = ERAU_MAT,
    show: bool = True,
):
    """Plot median-filtered ERAU PIP D2 signals."""
    fig, ax = plt.subplots(figsize=(10, 4))

    for label, (time_since_tg_s, signal) in processed_erau_signals(mat_path).items():
        ax.plot(
            time_since_tg_s,
            signal,
            color=ROCKET_COLORS[label[:-2]],
            linewidth=1.2,
            label=label,
        )

    ax.set_title("ERAU PIP data")
    ax.set_xlabel("Time since TG (s)")
    ax.set_ylabel("Current (nA)")
    ax.legend()
    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax


if __name__ == "__main__":
    export_erau_npz()
    plot_erau(show=True)
