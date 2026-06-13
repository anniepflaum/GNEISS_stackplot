from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parents[1] / "data"
APP_DATA_DIR = DATA_DIR / "app_data"
SOURCE_DATA_DIR = DATA_DIR / "source_data"
PIP_VOFF_NPZ = APP_DATA_DIR / "pip3_0_voff_data.npz"
PIP_VOFF_PLOT_PNG = SOURCE_DATA_DIR / "pip3_0_voff_vs_time.png"
TG_TO_MAGLAT_CSV = APP_DATA_DIR / "tg_to_maglat.csv"
ROCKETS = ("397", "398")
PIP_NAME = "PIP3"
GAIN = "0"
SCATTER_KWARGS = {
    "color": "green",
    "s": 1,
    "alpha": 0.2,
}
X_LIMITS_S = (100, 550)
Y_LIMITS_V = (-3, 1)


def pip_voff_paths(rocket: str) -> tuple[Path, Path]:
    stem = f"{rocket}_{PIP_NAME}_{GAIN}_SigmoidFit"
    return (
        SOURCE_DATA_DIR / f"{stem}_time.npy",
        SOURCE_DATA_DIR / f"{stem}_Voff.npy",
    )


def load_pip_voff(rocket: str):
    """Load notebook-matched positive flight time samples and Voff values."""
    time_path, voff_path = pip_voff_paths(rocket)
    time_s = np.load(time_path)
    voff = np.load(voff_path)
    positive_time_s = time_s[time_s > 0]

    if len(positive_time_s) != len(voff):
        raise ValueError(
            f"{rocket} positive time sample count does not match Voff: "
            f"{len(positive_time_s)} != {len(voff)}"
        )

    return positive_time_s, voff


def load_maglat_mapping(csv_path: str | Path = TG_TO_MAGLAT_CSV) -> dict:
    """Load TG-time to magnetic-latitude mappings for both rockets."""
    data = pd.read_csv(csv_path)
    mappings = {}

    for rocket in ROCKETS:
        column = f"{rocket}_magnetic_lat_deg"
        valid = data[column].notna()
        mappings[rocket] = (
            data.loc[valid, "time_since_TG_s"].to_numpy(),
            data.loc[valid, column].to_numpy(),
        )

    return mappings


def interpolate_maglat(time_since_tg_s, rocket: str, mappings=None):
    """Interpolate one rocket's magnetic latitude onto PIP Voff time samples."""
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


def export_pip_voff_npz(
    output_path: str | Path = PIP_VOFF_NPZ,
) -> Path:
    """Export low-gain PIP3 Voff data with magnetic latitude for the app."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    series = []
    maglat_mappings = load_maglat_mapping()

    for rocket in ROCKETS:
        time_s, voff = load_pip_voff(rocket)
        time_key = f"{rocket}_time_since_TG_s"
        maglat_key = f"{rocket}_magnetic_lat_deg"
        value_key = f"{rocket}_voff"
        arrays[time_key] = time_s
        arrays[maglat_key] = interpolate_maglat(
            time_s,
            rocket,
            mappings=maglat_mappings,
        )
        arrays[value_key] = voff
        time_path, voff_path = pip_voff_paths(rocket)
        series.append(
            {
                "rocket": rocket,
                "label": f"{rocket} PIP3 Voff",
                "time_key": time_key,
                "maglat_key": maglat_key,
                "value_key": value_key,
                "units": "V",
                "source_data_file": {
                    "time": time_path.name,
                    "voff": voff_path.name,
                },
                "plot_title": f"{rocket} Low Gain Voff vs time",
            }
        )

    metadata = {
        "pip": PIP_NAME,
        "gain": GAIN,
        "maglat_mapping_file": TG_TO_MAGLAT_CSV.name,
        "x_limits_s": X_LIMITS_S,
        "y_limits_v": Y_LIMITS_V,
        "processing": "uses positive time samples to match the notebook plot",
        "series": series,
    }
    np.savez_compressed(
        output_path,
        **arrays,
        metadata_json=np.array(json.dumps(metadata)),
    )
    return output_path


def plot_pip_voff(
    output_path: str | Path | None = PIP_VOFF_PLOT_PNG,
    show: bool = True,
):
    """Plot low-gain PIP3 Voff versus flight time for both rockets."""
    fig, axes = plt.subplots(len(ROCKETS), 1, figsize=(8, 6))

    for ax, rocket in zip(axes, ROCKETS):
        time_s, voff = load_pip_voff(rocket)
        ax.scatter(time_s, voff, **SCATTER_KWARGS)
        ax.set_xlim(*X_LIMITS_S)
        ax.set_ylim(*Y_LIMITS_V)
        ax.set_ylabel("Voff")
        ax.set_xlabel("flight time")
        ax.set_title(f"{rocket} Low Gain Voff vs time")

    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)

    if show:
        plt.show()

    return fig, axes


if __name__ == "__main__":
    export_pip_voff_npz()
    plot_pip_voff(show=True)
