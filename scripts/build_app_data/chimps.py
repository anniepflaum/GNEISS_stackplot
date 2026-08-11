from __future__ import annotations

from pathlib import Path

from data_paths import app_data_dir, source_data_dir

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

try:
    from .hdf5_io import write_hdf5
except ImportError:
    from hdf5_io import write_hdf5


SCRIPT_DIR = Path(__file__).resolve().parent
APP_DATA_DIR = app_data_dir()
SOURCE_DATA_DIR = source_data_dir()
CHIMPS_MAT = SOURCE_DATA_DIR / "GNEISS_397_CHIMPS_Down_V2.mat"
CHIMPS_H5 = APP_DATA_DIR / "chimps_397_downgoing_data.h5"
CHIMPS_SPECTROGRAM_JPG = SOURCE_DATA_DIR / "CHIMPS_GNEISS_Downgoing2_recreated.jpg"
CHIMPS_LINEPLOT_JPG = SOURCE_DATA_DIR / "GNEISS_CHIMPS_down_lineplot_recreated.jpg"
TIME_VARIABLE = "Final_Time"
ENERGY_VARIABLE = "Final_Energy"
FLUX_VARIABLE = "Final_Downgoing_Flux"
LINEPLOT_VARIABLE = "Final_lineplot_down"
ROCKET = "397"
PITCH_ANGLE_DESCRIPTION = "summed over the <30 pitch angles"
ENERGY_RANGE_DESCRIPTION = "~1 to ~12 keV"
LOG_COUNTS_LIMITS = (0.0, 4.5)
SPECTROGRAM_TIME_LIMITS_S = (160, 525)
SPECTROGRAM_LOG_ENERGY_LIMITS = (2.25, 4.16)
LINEPLOT_TIME_LIMITS_S = (150, 550)
LINEPLOT_COUNT_LIMITS = (0, 30000)


def load_chimps_data(mat_path: str | Path = CHIMPS_MAT) -> dict:
    """Load CHIMPS downgoing electron arrays from the MATLAB export."""
    mat_path = Path(mat_path)
    data = loadmat(
        mat_path,
        variable_names=[
            TIME_VARIABLE,
            ENERGY_VARIABLE,
            FLUX_VARIABLE,
            LINEPLOT_VARIABLE,
        ],
        squeeze_me=True,
    )
    time_s = np.asarray(data[TIME_VARIABLE], dtype=float).ravel()
    energy_eV = np.asarray(data[ENERGY_VARIABLE], dtype=float).ravel()
    counts = np.asarray(data[FLUX_VARIABLE], dtype=float)
    total_counts = np.asarray(data[LINEPLOT_VARIABLE], dtype=float).ravel()

    if counts.shape != (len(energy_eV), len(time_s)):
        raise ValueError(
            "CHIMPS flux shape does not match energy/time axes: "
            f"{counts.shape} != {(len(energy_eV), len(time_s))}"
        )
    if total_counts.shape != time_s.shape:
        raise ValueError(
            "CHIMPS lineplot shape does not match time axis: "
            f"{total_counts.shape} != {time_s.shape}"
        )

    order = np.argsort(time_s, kind="stable")
    time_s = time_s[order]
    counts = counts[:, order]
    total_counts = total_counts[order]
    time_s, unique_indices = np.unique(time_s, return_index=True)
    counts = counts[:, unique_indices]
    total_counts = total_counts[unique_indices]

    return {
        "time_since_TG_s": time_s,
        "energy_eV": energy_eV,
        "log10_energy_eV": np.log10(energy_eV),
        "counts": counts,
        "log10_counts": np.log10(np.clip(counts, 1.0, None)),
        "total_counts": total_counts,
    }


def energy_edges(log10_energy_eV):
    """Return pcolormesh edges for log-energy bin centers."""
    log10_energy_eV = np.asarray(log10_energy_eV, dtype=float)
    midpoints = (log10_energy_eV[:-1] + log10_energy_eV[1:]) / 2
    first_edge = log10_energy_eV[0] - (midpoints[0] - log10_energy_eV[0])
    last_edge = log10_energy_eV[-1] + (log10_energy_eV[-1] - midpoints[-1])
    return np.concatenate([[first_edge], midpoints, [last_edge]])


def time_edges(time_s):
    """Return pcolormesh edges for time sample centers."""
    time_s = np.asarray(time_s, dtype=float)
    midpoints = (time_s[:-1] + time_s[1:]) / 2
    first_edge = time_s[0] - (midpoints[0] - time_s[0])
    last_edge = time_s[-1] + (time_s[-1] - midpoints[-1])
    return np.concatenate([[first_edge], midpoints, [last_edge]])


def export_chimps_hdf5(
    output_path: str | Path = CHIMPS_H5,
    mat_path: str | Path = CHIMPS_MAT,
) -> Path:
    """Export CHIMPS 397 downgoing electron arrays for the stackplot app."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    chimps = load_chimps_data(mat_path)
    metadata = {
        "source_data_file": Path(mat_path).name,
        "rocket": ROCKET,
        "time_variable": TIME_VARIABLE,
        "energy_variable": ENERGY_VARIABLE,
        "flux_variable": FLUX_VARIABLE,
        "lineplot_variable": LINEPLOT_VARIABLE,
        "pitch_angle_description": PITCH_ANGLE_DESCRIPTION,
        "energy_range_description": ENERGY_RANGE_DESCRIPTION,
        "log10_counts_limits": LOG_COUNTS_LIMITS,
        "processing": (
            "time samples sorted and duplicate times reduced to the first sample; "
            "log10_counts computed as log10(max(counts, 1))"
        ),
    }
    return write_hdf5(
        output_path,
        chimps,
        metadata_json=metadata,
    )


def plot_chimps_spectrogram(
    output_path: str | Path | None = CHIMPS_SPECTROGRAM_JPG,
    mat_path: str | Path = CHIMPS_MAT,
    show: bool = True,
):
    """Plot the CHIMPS spectrogram with the reference labels and limits."""
    chimps = load_chimps_data(mat_path)
    fig, ax = plt.subplots(figsize=(12, 6))
    mesh = ax.pcolormesh(
        time_edges(chimps["time_since_TG_s"]),
        energy_edges(chimps["log10_energy_eV"]),
        chimps["log10_counts"],
        shading="auto",
        cmap="jet",
        vmin=LOG_COUNTS_LIMITS[0],
        vmax=LOG_COUNTS_LIMITS[1],
    )
    cbar = fig.colorbar(mesh, ax=ax, pad=0.01)
    cbar.set_label("Log (Counts)")
    ax.set_title(
        "CHIMPS: GNEISS: 36.397  Downgoing electrons "
        "(Summed over the <30 pitch angles)",
        fontweight="bold",
    )
    ax.set_xlabel("Time since TG (s)")
    ax.set_ylabel("Log10(Energy in eV)")
    ax.set_xlim(*SPECTROGRAM_TIME_LIMITS_S)
    ax.set_ylim(*SPECTROGRAM_LOG_ENERGY_LIMITS)
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)

    if show:
        plt.show()

    return fig, ax


def plot_chimps_lineplot(
    output_path: str | Path | None = CHIMPS_LINEPLOT_JPG,
    mat_path: str | Path = CHIMPS_MAT,
    show: bool = True,
):
    """Plot CHIMPS total counts with the reference labels and limits."""
    chimps = load_chimps_data(mat_path)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        chimps["time_since_TG_s"],
        chimps["total_counts"],
        color="black",
        linewidth=1.0,
    )
    ax.set_title(
        "CHIMPS: GNEISS: 36.397  Downgoing electrons "
        "(Total Counts, Summed over the <30 pitch angles and ~1 to ~12 keV)",
        fontweight="bold",
    )
    ax.set_xlabel("Time since TG (s)")
    ax.set_ylabel("Total Counts")
    ax.set_xlim(*LINEPLOT_TIME_LIMITS_S)
    ax.set_ylim(*LINEPLOT_COUNT_LIMITS)
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)

    if show:
        plt.show()

    return fig, ax


if __name__ == "__main__":
    export_chimps_hdf5()
    plot_chimps_spectrogram(show=True)
    plot_chimps_lineplot(show=True)
