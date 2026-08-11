from __future__ import annotations

import csv
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
EXB_CSVS = {
    "397": SOURCE_DATA_DIR / "exb_v2_397.csv",
    "398": SOURCE_DATA_DIR / "exb_v2_398.csv",
}
TG_TO_MAGLAT_CSV = APP_DATA_DIR / "tg_to_maglat.csv"
EXB_H5 = APP_DATA_DIR / "exb_components_data.h5"
EXB_TG_PLOT_PNG = SOURCE_DATA_DIR / "exb_components_time_since_TG.png"
EXB_MAGLAT_PLOT_PNG = SOURCE_DATA_DIR / "exb_components_maglat.png"
ROCKET_COLORS = {
    "397": "tab:blue",
    "398": "tab:orange",
}
T0_TG_OFFSET_S = {
    "397": 0.3167,
    "398": 30.4242,
}
TG_X_LIMITS_S = (0.0, 588.0)
Y_LIMIT_TIME_RANGE_S = (90.0, 520.0)
ROBUST_Y_PERCENTILES = (0.1, 99.99)
COMPONENTS = (
    {"key": "east", "column": 1, "label": "(ExB)/B^2 east"},
    {"key": "north", "column": 2, "label": "(ExB)/B^2 north"},
    {"key": "up", "column": 3, "label": "(ExB)/B^2 up"},
)


def load_exb_component(
    csv_path: str | Path,
    component_column: int,
    rocket: str,
):
    """Load one ExB component and convert the file's local t0 to TG time."""
    data = np.loadtxt(Path(csv_path), delimiter=",", comments="#")
    local_time_s = data[:, 0]
    component = data[:, component_column]
    time_since_tg_s = local_time_s + T0_TG_OFFSET_S[rocket]
    return time_since_tg_s, component


def load_maglat_mapping(
    csv_path: str | Path = TG_TO_MAGLAT_CSV,
):
    """Load TG-time to magnetic-latitude mappings for both rockets."""
    mappings = {}

    with Path(csv_path).open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)

    for rocket in ("397", "398"):
        time_s = []
        maglat_deg = []
        maglat_column = f"{rocket}_magnetic_lat_deg"

        for row in rows:
            if row[maglat_column]:
                time_s.append(float(row["time_since_TG_s"]))
                maglat_deg.append(float(row[maglat_column]))

        mappings[rocket] = (np.array(time_s), np.array(maglat_deg))

    return mappings


def interpolate_maglat(time_since_tg_s, rocket: str, mappings=None):
    """Interpolate one rocket's magnetic latitude onto TG time samples."""
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


def robust_limits(values, percentiles=ROBUST_Y_PERCENTILES):
    """Return plot limits that ignore extreme display outliers."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return None

    low, high = np.nanpercentile(values, percentiles)
    if not np.isfinite(low) or not np.isfinite(high):
        return None
    if low == high:
        margin = abs(low) * 0.05 if low else 1.0
        return low - margin, high + margin

    margin = 0.05 * (high - low)
    return low - margin, high + margin


def values_within_x_limits(x_values, y_values, x_limits=None):
    """Return finite y values that fall inside the displayed x range."""
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    valid = np.isfinite(x_values) & np.isfinite(y_values)

    if x_limits is not None:
        x_min, x_max = x_limits
        valid &= (x_values >= x_min) & (x_values <= x_max)

    return y_values[valid]


def component_y_limits(component_spec):
    """Return shared y-limits for one component using the TG display window."""
    component_values = []

    for rocket in ("397", "398"):
        time_since_tg_s, component = load_exb_component(
            EXB_CSVS[rocket],
            component_column=component_spec["column"],
            rocket=rocket,
        )
        component_values.append(
            values_within_x_limits(
                time_since_tg_s,
                component,
                x_limits=Y_LIMIT_TIME_RANGE_S,
            )
        )

    return robust_limits(np.concatenate(component_values))


def exb_series_key(rocket: str, component_key: str, suffix: str):
    """Build a stable HDF5 dataset key for one ExB component series."""
    return f"{rocket}_exb_{component_key}_{suffix}"


def export_exb_hdf5(
    output_path: str | Path = EXB_H5,
) -> Path:
    """Export full-resolution ExB component series for the stackplot app."""
    arrays = {}
    series = []
    maglat_mappings = load_maglat_mapping()
    y_limits = {
        component_spec["key"]: list(component_y_limits(component_spec))
        for component_spec in COMPONENTS
    }

    for rocket in ("397", "398"):
        for component_spec in COMPONENTS:
            time_since_tg_s, component = load_exb_component(
                EXB_CSVS[rocket],
                component_column=component_spec["column"],
                rocket=rocket,
            )
            time_key = exb_series_key(rocket, component_spec["key"], "time_since_TG_s")
            maglat_key = exb_series_key(rocket, component_spec["key"], "magnetic_lat_deg")
            value_key = exb_series_key(rocket, component_spec["key"], "value")
            arrays[time_key] = time_since_tg_s
            arrays[maglat_key] = interpolate_maglat(
                time_since_tg_s,
                rocket,
                mappings=maglat_mappings,
            )
            arrays[value_key] = component
            series.append(
                {
                    "rocket": rocket,
                    "component": component_spec["key"],
                    "label": f"{rocket} {component_spec['label']}",
                    "time_key": time_key,
                    "maglat_key": maglat_key,
                    "value_key": value_key,
                    "units": "m/s",
                    "color": ROCKET_COLORS[rocket],
                    "source_data_file": EXB_CSVS[rocket].name,
                }
            )

    metadata = {
        "source_data_file": {
            rocket: path.name for rocket, path in EXB_CSVS.items()
        },
        "maglat_mapping_file": TG_TO_MAGLAT_CSV.name,
        "t0_tg_offset_s": T0_TG_OFFSET_S,
        "resolution": "full source resolution",
        "display_x_limits_s": TG_X_LIMITS_S,
        "y_limit_time_range_s": Y_LIMIT_TIME_RANGE_S,
        "robust_y_percentiles": ROBUST_Y_PERCENTILES,
        "y_limits": y_limits,
        "series": series,
    }
    return write_hdf5(
        output_path,
        arrays,
        metadata_json=metadata,
    )


def plot_exb_components(
    output_path: str | Path | None = EXB_TG_PLOT_PNG,
    x_axis: str = "time_since_TG",
    show: bool = True,
):
    """Plot ExB east, north, and up components for both rockets."""
    if x_axis not in {"time_since_TG", "maglat"}:
        raise ValueError("x_axis must be 'time_since_TG' or 'maglat'")

    if output_path is None and x_axis == "maglat":
        output_path = EXB_MAGLAT_PLOT_PNG

    maglat_mappings = load_maglat_mapping() if x_axis == "maglat" else None
    x_axis_label = "magnetic latitude" if x_axis == "maglat" else "time"
    x_limits = TG_X_LIMITS_S if x_axis == "time_since_TG" else None
    shared_y_limits = {
        component_spec["key"]: component_y_limits(component_spec)
        for component_spec in COMPONENTS
    }
    fig, axes = plt.subplots(len(COMPONENTS), 1, figsize=(10, 9), sharex=True)

    for ax, component_spec in zip(axes, COMPONENTS):
        for rocket in ("397", "398"):
            time_since_tg_s, component = load_exb_component(
                EXB_CSVS[rocket],
                component_column=component_spec["column"],
                rocket=rocket,
            )
            x_values = (
                interpolate_maglat(time_since_tg_s, rocket, mappings=maglat_mappings)
                if x_axis == "maglat"
                else time_since_tg_s
            )
            ax.plot(
                x_values,
                component,
                color=ROCKET_COLORS[rocket],
                linewidth=0.8,
                label=f"{rocket} {component_spec['label']}",
            )

        ax.set_title(f"{component_spec['label']} vs {x_axis_label}")
        ax.set_ylabel(f"{component_spec['label']} (m/s)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", framealpha=0.95)
        if x_limits is not None:
            ax.set_xlim(*x_limits)

        y_limits = shared_y_limits[component_spec["key"]]
        if y_limits is not None:
            ax.set_ylim(*y_limits)

    if x_axis == "maglat":
        axes[-1].set_xlabel("Magnetic latitude (deg)")
    else:
        axes[-1].set_xlabel("Time since TG (s)")
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)

    if show:
        plt.show()

    return fig, axes


if __name__ == "__main__":
    export_exb_hdf5()
    plot_exb_components(
        output_path=EXB_TG_PLOT_PNG,
        x_axis="time_since_TG",
        show=True,
    )
    plot_exb_components(
        output_path=EXB_MAGLAT_PLOT_PNG,
        x_axis="maglat",
        show=True,
    )
