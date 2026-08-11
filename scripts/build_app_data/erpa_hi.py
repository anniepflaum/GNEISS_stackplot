from __future__ import annotations

from pathlib import Path

from data_paths import app_data_dir, source_data_dir

import matplotlib.pyplot as plt
import numpy as np

try:
    from .hdf5_io import write_hdf5
except ImportError:
    from hdf5_io import write_hdf5


SCRIPT_DIR = Path(__file__).resolve().parent
APP_DATA_DIR = app_data_dir()
SOURCE_DATA_DIR = source_data_dir()
ERPA_HI_CSVS = {
    "397": SOURCE_DATA_DIR / "fwd397hi.csv",
    "398": SOURCE_DATA_DIR / "fwd398hi.csv",
}
ERPA_HI_H5 = APP_DATA_DIR / "erpa_hi_data.h5"
TG_TO_MAGLAT_CSV = APP_DATA_DIR / "tg_to_maglat.csv"
T0_TG_OFFSET_S = {
    "397": 0.0,
    "398": 30.0,
}
ROCKET_COLORS = {
    "397": "tab:blue",
    "398": "tab:orange",
}
TG_X_LIMITS_S = (0.0, 588.0)


def load_erpa_hi(
    csv_path: str | Path,
    rocket: str,
):
    """Load ERPA hi magnitude and convert local t0 time to TG time."""
    data = np.genfromtxt(
        Path(csv_path),
        delimiter=",",
        missing_values="?",
        filling_values=np.nan,
    )
    time_since_tg_s = data[:, 0] + T0_TG_OFFSET_S[rocket]
    hi = data[:, 1]
    return time_since_tg_s, hi


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


def export_erpa_hi_hdf5(
    output_path: str | Path = ERPA_HI_H5,
):
    """Export full-resolution TG-aligned ERPA hi arrays and metadata."""
    arrays = {}
    series = []
    maglat_mappings = load_maglat_mapping()

    for rocket, csv_path in ERPA_HI_CSVS.items():
        time_key = f"{rocket}_time_since_TG_s"
        maglat_key = f"{rocket}_magnetic_lat_deg"
        value_key = f"{rocket}_hi"
        time_since_tg_s, hi = load_erpa_hi(csv_path, rocket)
        arrays[time_key] = time_since_tg_s
        arrays[maglat_key] = interpolate_maglat(
            time_since_tg_s,
            rocket,
            mappings=maglat_mappings,
        )
        arrays[value_key] = hi
        series.append(
            {
                "rocket": rocket,
                "time_key": time_key,
                "maglat_key": maglat_key,
                "value_key": value_key,
                "label": f"{rocket} hi",
                "units": "counts",
                "color": ROCKET_COLORS[rocket],
            }
        )

    metadata = {
        "source_data_file": {
            rocket: path.name for rocket, path in ERPA_HI_CSVS.items()
        },
        "maglat_mapping_file": TG_TO_MAGLAT_CSV.name,
        "resolution": "full source resolution",
        "series": series,
    }
    return write_hdf5(
        output_path,
        arrays,
        metadata_json=metadata,
    )


def plot_erpa_hi(
    output_path: str | Path | None = None,
    show: bool = True,
):
    """Plot ERPA hi magnitude for rockets 397 and 398 on the TG time axis."""
    fig, ax = plt.subplots(figsize=(10, 4))

    for rocket, csv_path in ERPA_HI_CSVS.items():
        time_since_tg_s, hi = load_erpa_hi(csv_path, rocket)
        ax.plot(
            time_since_tg_s,
            hi,
            color=ROCKET_COLORS[rocket],
            linewidth=1.2,
            label=f"{rocket} hi",
        )

    ax.set_title("ERPA hi")
    ax.set_xlabel("Time since TG (s)")
    ax.set_ylabel("hi (counts)")
    ax.set_xlim(*TG_X_LIMITS_S)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", framealpha=0.95)
    fig.tight_layout()


    '''
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
    '''

    if show:
        plt.show()

    return fig, ax


if __name__ == "__main__":
    export_erpa_hi_hdf5()
    plot_erpa_hi(show=True)
