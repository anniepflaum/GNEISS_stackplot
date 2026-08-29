from __future__ import annotations

import csv
from pathlib import Path

from data_paths import app_data_dir, source_data_dir

import h5py
import matplotlib.pyplot as plt
import numpy as np

try:
    from .hdf5_io import write_hdf5
except ImportError:
    from hdf5_io import write_hdf5


SCRIPT_DIR = Path(__file__).resolve().parent
APP_DATA_DIR = app_data_dir()
SOURCE_DATA_DIR = source_data_dir()
B_FIELD_CSVS = {
    "397": SOURCE_DATA_DIR / "b_despun_v2_397.csv",
    "398": SOURCE_DATA_DIR / "b_despun_398.csv",
}
E_FIELD_CSVS = {
    "397": SOURCE_DATA_DIR / "e_despun_subtracted_397_july30.csv",
    "398": SOURCE_DATA_DIR / "e_despun_subtracted_398_july30.csv",
}
BRIGHTNESS_H5 = (
    SOURCE_DATA_DIR
    / "brightness_vs_time_20260210_101900_102848_step0p05.h5"
)
B_E_STACKPLOT_DATA_H5 = APP_DATA_DIR / "b_e_field_components_data.h5"
TG_TO_MAGLAT_CSV = APP_DATA_DIR / "tg_to_maglat.csv"
B_E_TG_PLOT_PNG = SOURCE_DATA_DIR / "b_e_field_components_time_since_TG.png"
B_E_MAGLAT_PLOT_PNG = SOURCE_DATA_DIR / "b_e_field_components_maglat.png"
ROCKET_COLORS = {
    "397": "tab:blue",
    "398": "tab:orange",
}
B_ROCKET_COLORS = {
    "397": "#8ecae6",
    "398": "#ffb36b",
}
BRIGHTNESS_ALTITUDES_KM = (95, 100, 105, 110)
B_E_T0_TG_OFFSET_S = {
    "397": 0.3167,
    "398": 30.4242,
}
B_E_LOCAL_TIME_RANGE_S = (90.0, 520.0)
TG_X_LIMITS_S = (0.0, 588.0)
B_E_COMPONENT_SPECS = (
    {
        "panel": "b_north_e_east",
        "source": "b",
        "component": "B_north",
        "component_column": 2,
        "units": "nT",
        "line_style": "solid",
    },
    {
        "panel": "b_north_e_east",
        "source": "e",
        "component": "E_east",
        "component_column": 1,
        "units": "V/m",
        "line_style": "solid",
    },
    {
        "panel": "b_east_e_north",
        "source": "b",
        "component": "B_east",
        "component_column": 1,
        "units": "nT",
        "line_style": "solid",
    },
    {
        "panel": "b_east_e_north",
        "source": "e",
        "component": "E_north",
        "component_column": 2,
        "units": "V/m",
        "line_style": "solid",
    },
)


def load_component(
    csv_path: str | Path,
    component_column: int,
    rocket: str | None = None,
):
    """Load TG time and one component column from a comma-delimited data file."""
    data = np.loadtxt(Path(csv_path), delimiter=",", comments="#")
    local_time_s = data[:, 0]
    component = data[:, component_column]

    if rocket is not None:
        start_s, end_s = B_E_LOCAL_TIME_RANGE_S
        keep = (local_time_s >= start_s) & (local_time_s <= end_s)
        local_time_s = local_time_s[keep]
        component = component[keep]

    if rocket is not None:
        time_s = local_time_s + B_E_T0_TG_OFFSET_S[rocket]
    else:
        time_s = local_time_s

    return time_s, component


def load_maglat_mapping(
    csv_path: str | Path = TG_TO_MAGLAT_CSV,
):
    """Load TG-to-magnetic-latitude mappings for both rockets."""
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
    """Interpolate one rocket's magnetic latitude onto a TG time array."""
    if mappings is None:
        mappings = load_maglat_mapping()

    mapping_time_s, mapping_maglat_deg = mappings[rocket]
    return np.interp(time_since_tg_s, mapping_time_s, mapping_maglat_deg)


def put_left_axis_above_twin(left_ax, right_ax):
    """Draw left-axis artists, including legends, above the twin y-axis."""
    left_ax.set_zorder(right_ax.get_zorder() + 1)
    left_ax.patch.set_visible(False)


def b_e_series_key(panel: str, rocket: str, component: str, suffix: str):
    """Build a stable HDF5 dataset key for one plotted data series."""
    return f"{panel}_{rocket}_{component}_{suffix}"


def build_b_e_field_plot_data():
    """Build the filtered, TG-aligned B/E arrays used by the stackplot panels."""
    arrays = {}
    metadata = []
    maglat_mappings = load_maglat_mapping()

    for rocket in ("397", "398"):
        for spec in B_E_COMPONENT_SPECS:
            if spec["source"] == "b":
                source_csv = B_FIELD_CSVS[rocket]
            else:
                source_csv = E_FIELD_CSVS[rocket]

            time_s, values = load_component(
                source_csv,
                component_column=spec["component_column"],
                rocket=rocket,
            )
            key_prefix = b_e_series_key(
                spec["panel"],
                rocket,
                spec["component"],
                "",
            ).rstrip("_")

            arrays[f"{key_prefix}_time_since_TG_s"] = time_s
            arrays[f"{key_prefix}_magnetic_lat_deg"] = interpolate_maglat(
                time_s,
                rocket,
                mappings=maglat_mappings,
            )
            arrays[f"{key_prefix}_value"] = values
            metadata.append(
                {
                    "panel": spec["panel"],
                    "rocket": rocket,
                    "source": spec["source"],
                    "component": spec["component"],
                    "time_key": f"{key_prefix}_time_since_TG_s",
                    "maglat_key": f"{key_prefix}_magnetic_lat_deg",
                    "value_key": f"{key_prefix}_value",
                    "units": spec["units"],
                    "color": (
                        B_ROCKET_COLORS[rocket]
                        if spec["source"] == "b"
                        else ROCKET_COLORS[rocket]
                    ),
                    "line_style": spec["line_style"],
                    "source_data_file": str(source_csv),
                    "maglat_mapping_file": str(TG_TO_MAGLAT_CSV),
                }
            )

    return arrays, metadata


def export_b_e_field_plot_data_hdf5(
    output_path: str | Path = B_E_STACKPLOT_DATA_H5,
):
    """Write full-resolution, TG-aligned B/E data for the stackplot panels."""
    arrays, metadata = build_b_e_field_plot_data()
    provenance = {
        "b_field_files": {rocket: str(path) for rocket, path in B_FIELD_CSVS.items()},
        "e_field_files": {rocket: str(path) for rocket, path in E_FIELD_CSVS.items()},
        "maglat_mapping_file": str(TG_TO_MAGLAT_CSV),
        "resolution": "full source resolution",
    }
    return write_hdf5(
        output_path,
        arrays,
        metadata_json=metadata,
        provenance_json=provenance,
    )


def export_b_e_field_plot_data_csv(output_path: str | Path):
    """Write the filtered, TG-aligned B/E data as a long CSV file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = (
        "panel",
        "rocket",
        "source",
        "component",
        "time_since_TG_s",
        "magnetic_lat_deg",
        "value",
        "units",
        "color",
        "line_style",
        "source_data_file",
        "maglat_mapping_file",
    )

    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        arrays, metadata = build_b_e_field_plot_data()

        for series in metadata:
            time_s = arrays[series["time_key"]]
            maglat_deg = arrays[series["maglat_key"]]
            values = arrays[series["value_key"]]

            for time_value, maglat_value, data_value in zip(time_s, maglat_deg, values):
                writer.writerow(
                    {
                        "panel": series["panel"],
                        "rocket": series["rocket"],
                        "source": series["source"],
                        "component": series["component"],
                        "time_since_TG_s": f"{time_value:.9f}",
                        "magnetic_lat_deg": f"{maglat_value:.9f}",
                        "value": f"{data_value:.12g}",
                        "units": series["units"],
                        "color": series["color"],
                        "line_style": series["line_style"],
                        "source_data_file": series["source_data_file"],
                        "maglat_mapping_file": series["maglat_mapping_file"],
                    }
                )

    return output_path


def load_brightness(
    brightness_h5_path: str | Path = BRIGHTNESS_H5,
    rocket: str = "397",
    altitude_km: int | None = None,
):
    """Load brightness versus TG for one rocket."""
    brightness_h5_path = Path(brightness_h5_path)
    with h5py.File(brightness_h5_path, "r") as source:
        if source.attrs.get("brightness_units") != "Rayleighs":
            raise ValueError(f"{brightness_h5_path} is not Rayleigh-calibrated")
        time_s = np.asarray(source["time_since_tg_s"], dtype=float)
        brightness_group = source[f"rockets/{rocket}/brightness"]
        if altitude_km is None:
            dataset_names = [
                f"{altitude}_km"
                for altitude in BRIGHTNESS_ALTITUDES_KM
                if f"{altitude}_km" in brightness_group
            ]
        else:
            dataset_names = [f"{altitude_km}_km"]
        if not dataset_names:
            raise KeyError(f"No requested brightness altitude is available for rocket {rocket}")
        brightness_arrays = [
            np.asarray(brightness_group[name], dtype=float)
            for name in dataset_names
        ]

    brightness_stack = np.vstack(brightness_arrays)
    finite_count = np.sum(np.isfinite(brightness_stack), axis=0)
    brightness = np.full(time_s.shape, np.nan, dtype=float)
    np.divide(
        np.nansum(brightness_stack, axis=0),
        finite_count,
        out=brightness,
        where=finite_count > 0,
    )
    valid = np.isfinite(time_s) & np.isfinite(brightness)
    return time_s[valid], brightness[valid]


def plot_brightness(
    brightness_h5_path: str | Path = BRIGHTNESS_H5,
    altitude_km: int | None = None,
    output_path: str | Path | None = None,
    show: bool = True,
):
    """Plot brightness for rockets 397 and 398."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for rocket in ("397", "398"):
        time_s, brightness = load_brightness(
            brightness_h5_path=brightness_h5_path,
            rocket=rocket,
            altitude_km=altitude_km,
        )
        ax.plot(
            time_s,
            brightness,
            color=ROCKET_COLORS[rocket],
            linewidth=1.2,
            label=f"{rocket} brightness",
        )

    if altitude_km is None:
        title = "Trajectory brightness"
        ylabel = "Brightness (mean of 95, 100, 105, 110 km)"
    else:
        title = f"Trajectory brightness at {altitude_km} km"
        ylabel = "Brightness"

    ax.set_title(title)
    ax.set_xlabel("TG (s)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", framealpha=0.95)
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)

    if show:
        plt.show()

    return fig, ax


def plot_b_north_e_east(
    output_path: str | Path | None = None,
    show: bool = True,
):
    """Plot B north and E east with source-specific colors for both rockets."""
    fig, b_ax = plt.subplots(figsize=(10, 5))
    e_ax = b_ax.twinx()
    put_left_axis_above_twin(b_ax, e_ax)

    b_lines = []
    e_lines = []

    for rocket in ("397", "398"):
        b_color = B_ROCKET_COLORS[rocket]
        e_color = ROCKET_COLORS[rocket]

        b_time_s, b_north_nt = load_component(
            B_FIELD_CSVS[rocket],
            component_column=2,
            rocket=rocket,
        )
        (b_line,) = b_ax.plot(
            b_time_s,
            b_north_nt,
            color=b_color,
            linestyle="-",
            linewidth=1.2,
            label=f"{rocket} B north",
        )
        b_lines.append(b_line)

        e_time_s, e_east_vpm = load_component(
            E_FIELD_CSVS[rocket],
            component_column=1,
            rocket=rocket,
        )
        (e_line,) = e_ax.plot(
            e_time_s,
            e_east_vpm,
            color=e_color,
            linestyle="-",
            linewidth=1.2,
            label=f"{rocket} E east",
        )
        e_lines.append(e_line)

    b_ax.set_title("B north and E east components")
    b_ax.set_xlabel("Time since TG (s)")
    b_ax.set_ylabel("B north (nT)")
    e_ax.set_ylabel("E east (V/m)")

    b_ax.set_xlim(*TG_X_LIMITS_S)
    b_ax.set_ylim(1500, 12000)
    e_ax.set_ylim(-0.05, 0.025)
    b_ax.grid(True, alpha=0.3)

    lines = b_lines + e_lines
    legend = b_ax.legend(
        lines,
        [line.get_label() for line in lines],
        loc="upper left",
        framealpha=0.95,
    )
    legend.set_zorder(10)
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)

    if show:
        plt.show()

    return fig, (b_ax, e_ax)


def plot_b_east_e_north(
    output_path: str | Path | None = None,
    show: bool = True,
):
    """Plot B east and E north with source-specific colors for both rockets."""
    fig, b_ax = plt.subplots(figsize=(10, 5))
    e_ax = b_ax.twinx()
    put_left_axis_above_twin(b_ax, e_ax)

    b_lines = []
    e_lines = []

    for rocket in ("397", "398"):
        b_color = B_ROCKET_COLORS[rocket]
        e_color = ROCKET_COLORS[rocket]

        b_time_s, b_east_nt = load_component(
            B_FIELD_CSVS[rocket],
            component_column=1,
            rocket=rocket,
        )
        (b_line,) = b_ax.plot(
            b_time_s,
            b_east_nt,
            color=b_color,
            linestyle="-",
            linewidth=1.2,
            label=f"{rocket} B east",
        )
        b_lines.append(b_line)

        e_time_s, e_north_vpm = load_component(
            E_FIELD_CSVS[rocket],
            component_column=2,
            rocket=rocket,
        )
        (e_line,) = e_ax.plot(
            e_time_s,
            e_north_vpm,
            color=e_color,
            linestyle="-",
            linewidth=1.2,
            label=f"{rocket} E north",
        )
        e_lines.append(e_line)

    b_ax.set_title("B east and E north components")
    b_ax.set_xlabel("Time since TG (s)")
    b_ax.set_ylabel("B east (nT)")
    e_ax.set_ylabel("E north (V/m)")

    b_ax.set_xlim(*TG_X_LIMITS_S)
    b_ax.set_ylim(1500, 12000)
    e_ax.set_ylim(-0.05, 0.025)
    b_ax.grid(True, alpha=0.3)

    lines = b_lines + e_lines
    legend = b_ax.legend(
        lines,
        [line.get_label() for line in lines],
        loc="upper left",
        framealpha=0.95,
    )
    legend.set_zorder(10)
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)

    if show:
        plt.show()

    return fig, (b_ax, e_ax)


def plot_b_e_field_components(
    output_path: str | Path | None = None,
    data_output_path: str | Path | None = B_E_STACKPLOT_DATA_H5,
    x_axis: str = "time_since_TG",
    show: bool = False,
):
    """Plot B/E component pairs in two stacked subplots."""
    if x_axis not in {"time_since_TG", "maglat"}:
        raise ValueError("x_axis must be 'time_since_TG' or 'maglat'")

    if data_output_path is not None:
        export_b_e_field_plot_data_hdf5(data_output_path)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    maglat_mappings = load_maglat_mapping() if x_axis == "maglat" else None

    top_b_ax = axes[0]
    top_e_ax = top_b_ax.twinx()
    bottom_b_ax = axes[1]
    bottom_e_ax = bottom_b_ax.twinx()
    put_left_axis_above_twin(top_b_ax, top_e_ax)
    put_left_axis_above_twin(bottom_b_ax, bottom_e_ax)

    subplot_specs = (
        {
            "b_ax": top_b_ax,
            "e_ax": top_e_ax,
            "title": "B north and E east components",
            "b_ylabel": "B north (nT)",
            "e_ylabel": "E east (V/m)",
            "b_column": 2,
            "e_column": 1,
            "b_label": "B north",
            "e_label": "E east",
        },
        {
            "b_ax": bottom_b_ax,
            "e_ax": bottom_e_ax,
            "title": "B east and E north components",
            "b_ylabel": "B east (nT)",
            "e_ylabel": "E north (V/m)",
            "b_column": 1,
            "e_column": 2,
            "b_label": "B east",
            "e_label": "E north",
        },
    )

    for spec in subplot_specs:
        b_lines = []
        e_lines = []

        for rocket in ("397", "398"):
            b_color = B_ROCKET_COLORS[rocket]
            e_color = ROCKET_COLORS[rocket]

            b_time_s, b_component = load_component(
                B_FIELD_CSVS[rocket],
                component_column=spec["b_column"],
                rocket=rocket,
            )
            b_x = (
                interpolate_maglat(b_time_s, rocket, mappings=maglat_mappings)
                if x_axis == "maglat"
                else b_time_s
            )
            (b_line,) = spec["b_ax"].plot(
                b_x,
                b_component,
                color=b_color,
                linestyle="-",
                linewidth=1.2,
                label=f"{rocket} {spec['b_label']}",
            )
            b_lines.append(b_line)

            e_time_s, e_component = load_component(
                E_FIELD_CSVS[rocket],
                component_column=spec["e_column"],
                rocket=rocket,
            )
            e_x = (
                interpolate_maglat(e_time_s, rocket, mappings=maglat_mappings)
                if x_axis == "maglat"
                else e_time_s
            )
            (e_line,) = spec["e_ax"].plot(
                e_x,
                e_component,
                color=e_color,
                linestyle="-",
                linewidth=1.2,
                label=f"{rocket} {spec['e_label']}",
            )
            e_lines.append(e_line)

        spec["b_ax"].set_title(spec["title"])
        spec["b_ax"].set_ylabel(spec["b_ylabel"])
        spec["e_ax"].set_ylabel(spec["e_ylabel"])
        if x_axis == "time_since_TG":
            spec["b_ax"].set_xlim(*TG_X_LIMITS_S)
        spec["b_ax"].set_ylim(1500, 12000)
        spec["e_ax"].set_ylim(-0.05, 0.025)
        spec["b_ax"].grid(True, alpha=0.3)

        lines = b_lines + e_lines
        legend = spec["b_ax"].legend(
            lines,
            [line.get_label() for line in lines],
            loc="upper left",
            framealpha=0.95,
        )
        legend.set_zorder(10)

    if x_axis == "maglat":
        bottom_b_ax.set_xlabel("Magnetic latitude (deg)")
    else:
        bottom_b_ax.set_xlabel("Time since TG (s)")
    fig.tight_layout()

    
    '''
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
    '''

    if show:
        plt.show()

    return fig, ((top_b_ax, top_e_ax), (bottom_b_ax, bottom_e_ax))


if __name__ == "__main__":
    plot_b_e_field_components(
        output_path=B_E_TG_PLOT_PNG,
        show=True,
    )
    plot_b_e_field_components(
        output_path=B_E_MAGLAT_PLOT_PNG,
        data_output_path=None,
        x_axis="maglat",
        show=True,
    )
