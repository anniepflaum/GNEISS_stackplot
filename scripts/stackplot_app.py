from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from dash import Dash, Input, Output, dcc, html
import plotly.graph_objects as go
from plotly.subplots import make_subplots


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
APP_DATA_DIR = DATA_DIR / "app_data"
B_E_NPZ = APP_DATA_DIR / "b_e_field_components_data.npz"
ERAU_NPZ = APP_DATA_DIR / "erau_signal_data.npz"
ERPA_HI_NPZ = APP_DATA_DIR / "erpa_hi_data.npz"
ERPA_TEMP_NPZ = APP_DATA_DIR / "erpa_temp_data.npz"
CHIMPS_NPZ = APP_DATA_DIR / "chimps_397_downgoing_data.npz"
PIP_VOFF_NPZ = APP_DATA_DIR / "pip3_0_voff_data.npz"
EXB_NPZ = APP_DATA_DIR / "exb_components_data.npz"
KEOGRAM_NPZ = APP_DATA_DIR / "trajectory_keogram_green_20260210_101900_102848.npz"
FOOTPOINT_BRIGHTNESS_NPZ = APP_DATA_DIR / "footpoint_brightness_data.npz"
TG_TO_MAGLAT_CSV = APP_DATA_DIR / "tg_to_maglat.csv"
TG_X_LIMITS_S = (0.0, 588.0)
TIME_STEP_S = 0.05
HEATMAP_TIME_STEP_S = 0.3
PANEL_HEIGHT_PX = 260
PANEL_GAP_PX = 42
PLOT_VERTICAL_MARGIN_PX = 55
ROCKET_COLORS = {
    "397": "tab:blue",
    "398": "tab:orange",
}
PLOTLY_COLORS = {
    "tab:blue": "#1f77b4",
    "tab:orange": "#ff7f0e",
}


def plotly_color(color: str) -> str:
    return PLOTLY_COLORS.get(color, color)


def browser_values(values):
    """Convert NumPy arrays to browser-safe JSON lists."""
    return np.asarray(values).tolist()


COMMON_TIME_S = np.round(
    np.arange(TG_X_LIMITS_S[0], TG_X_LIMITS_S[1] + TIME_STEP_S / 2, TIME_STEP_S),
    decimals=2,
)
HEATMAP_TIME_S = np.round(
    np.arange(
        TG_X_LIMITS_S[0],
        TG_X_LIMITS_S[1] + HEATMAP_TIME_STEP_S / 2,
        HEATMAP_TIME_STEP_S,
    ),
    decimals=2,
)


def common_time_subset(time_since_tg_s):
    """Return exact 0.05 s TG samples covered by an input time array."""
    finite_time = np.asarray(time_since_tg_s)[np.isfinite(time_since_tg_s)]
    if len(finite_time) == 0:
        return np.array([], dtype=float)

    return COMMON_TIME_S[
        (COMMON_TIME_S >= np.min(finite_time))
        & (COMMON_TIME_S <= np.max(finite_time))
    ]


def resample_series_to_common_time(time_since_tg_s, values):
    """Interpolate one series onto the shared 0.05 s TG grid."""
    time_since_tg_s = np.asarray(time_since_tg_s)
    values = np.asarray(values)
    valid = np.isfinite(time_since_tg_s) & np.isfinite(values)
    source_time_s = time_since_tg_s[valid]
    source_values = values[valid]
    target_time_s = common_time_subset(source_time_s)

    if len(source_time_s) == 0:
        return target_time_s, np.array([], dtype=float)

    return target_time_s, np.interp(target_time_s, source_time_s, source_values)


def resample_matrix_to_common_time(time_since_tg_s, values):
    """Interpolate each matrix row onto the shared 0.3 s heatmap grid."""
    time_since_tg_s = np.asarray(time_since_tg_s)
    values = np.asarray(values)
    finite_time = time_since_tg_s[np.isfinite(time_since_tg_s)]
    target_time_s = HEATMAP_TIME_S[
        (HEATMAP_TIME_S >= np.min(finite_time))
        & (HEATMAP_TIME_S <= np.max(finite_time))
    ]
    resampled = np.vstack(
        [np.interp(target_time_s, time_since_tg_s, row) for row in values]
    )
    return target_time_s, resampled


def load_maglat_mappings() -> dict:
    data = pd.read_csv(TG_TO_MAGLAT_CSV)
    mappings = {}

    for rocket in ("397", "398"):
        column = f"{rocket}_magnetic_lat_deg"
        valid = data[column].notna()
        mappings[rocket] = (
            data.loc[valid, "time_since_TG_s"].to_numpy(),
            data.loc[valid, column].to_numpy(),
        )

    return mappings


MAGLAT_MAPPINGS = load_maglat_mappings()


def interpolate_maglat(time_since_tg_s, rocket: str):
    mapping_time_s, mapping_maglat_deg = MAGLAT_MAPPINGS[rocket]
    return np.interp(
        time_since_tg_s,
        mapping_time_s,
        mapping_maglat_deg,
        left=np.nan,
        right=np.nan,
    )


def subplot_colorbar(row: int, row_count: int, title: str) -> dict:
    """Return a colorbar config sized to one subplot row."""
    return {
        "title": title,
        "len": 0.82 / row_count,
        "y": 1.0 - ((row - 0.5) / row_count),
        "yanchor": "middle",
        "x": 1.01,
        "thickness": 14,
    }


def add_panel_legend(fig: go.Figure, row: int, legend_items: list[dict]) -> None:
    """Draw a compact legend beside one subplot."""
    if not legend_items:
        return

    yaxis_name = "yaxis" if row == 1 else f"yaxis{2 * row - 1}"
    domain = getattr(fig.layout, yaxis_name).domain
    row_height = domain[1] - domain[0]
    y_start = domain[1] - 0.14 * row_height
    y_gap = min(0.035, 0.18 * row_height)

    for index, item in enumerate(legend_items):
        y = y_start - index * y_gap
        fig.add_shape(
            type="line",
            xref="paper",
            yref="paper",
            x0=1.045,
            x1=1.08,
            y0=y,
            y1=y,
            line={
                "color": item["color"],
                "width": 2,
                "dash": item["dash"],
            },
        )
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=1.09,
            y=y,
            text=item["name"],
            showarrow=False,
            xanchor="left",
            yanchor="middle",
            font={"size": 15},
            bgcolor="rgba(255,255,255,0.85)",
        )


def add_panel_title(fig: go.Figure, row: int, title: str) -> None:
    """Place a subplot title inside the upper-left corner of one panel."""
    xaxis_name = "xaxis" if row == 1 else f"xaxis{row}"
    yaxis_name = "yaxis" if row == 1 else f"yaxis{2 * row - 1}"
    x_domain = getattr(fig.layout, xaxis_name).domain
    y_domain = getattr(fig.layout, yaxis_name).domain
    row_height = y_domain[1] - y_domain[0]

    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=x_domain[0] + 0.012,
        y=y_domain[1] - 0.06 * row_height,
        text=title,
        showarrow=False,
        xanchor="left",
        yanchor="top",
        align="left",
        font={"size": 20},
        bgcolor="rgba(255,255,255,0.85)",
        borderpad=3,
    )


def source_file_names(source_data_file) -> list[str]:
    """Return unique source basenames from a metadata path collection."""
    if isinstance(source_data_file, dict):
        values = source_data_file.values()
    elif isinstance(source_data_file, (list, tuple)):
        values = source_data_file
    else:
        values = [source_data_file]

    def display_name(value: str) -> str:
        if " from " in value:
            return value
        return Path(value).name

    return list(dict.fromkeys(display_name(value) for value in values if value))


def panel_title(panel: dict, show_source_subtitle: bool = True) -> str:
    """Format a subplot title with source filenames on a second line."""
    title = f"<b>{panel['label']}</b>"
    source_files = panel.get("source_files", [])
    if not show_source_subtitle or not source_files:
        return title

    sources = ", ".join(source_files)
    return f"{title}<br><span style='font-size:12px'>Source data: {sources}</span>"


def load_b_e_panels() -> dict:
    npz = np.load(B_E_NPZ)
    metadata = json.loads(str(npz["metadata_json"]))
    provenance = json.loads(str(npz["provenance_json"]))
    source_files = source_file_names(
        list(provenance["b_field_files"].values())
        + list(provenance["e_field_files"].values())
    )
    panels = {}

    panel_titles = {
        "b_north_e_east": "B north and E east",
        "b_east_e_north": "B east and E north",
    }

    for series in metadata:
        panel_id = series["panel"]
        panels.setdefault(
            panel_id,
            {
                "label": panel_titles.get(panel_id, panel_id),
                "left_y_title": None,
                "right_y_title": None,
                "left_y_type": "linear",
                "left_y_range": [1500, 12000],
                "right_y_range": [-0.05, 0.025],
                "source_files": source_files,
                "traces": [],
            },
        )

        use_right_axis = series["source"] == "e"
        if use_right_axis:
            panels[panel_id]["right_y_title"] = f"{series['component']} ({series['units']})"
        else:
            panels[panel_id]["left_y_title"] = f"{series['component']} ({series['units']})"

        time_since_tg_s, y = resample_series_to_common_time(
            npz[series["time_key"]],
            npz[series["value_key"]],
        )
        magnetic_lat_deg = interpolate_maglat(time_since_tg_s, series["rocket"])
        panels[panel_id]["traces"].append(
            {
                "time_since_tg_s": time_since_tg_s,
                "magnetic_lat_deg": magnetic_lat_deg,
                "y": y,
                "name": f"{series['rocket']} {series['component']}",
                "color": plotly_color(series["color"]),
                "dash": "dash" if series["line_style"] == "dashed" else "solid",
                "secondary_y": use_right_axis,
                "type": "scatter",
            }
        )

    return panels


def load_footpoint_brightness_panel(altitude_km: int = 110) -> dict:
    data = np.load(FOOTPOINT_BRIGHTNESS_NPZ)
    metadata = json.loads(str(data["metadata_json"]))
    all_time_since_tg_s = data["time_since_TG_s"]
    traces = []

    for rocket in ("397", "398"):
        key = f"{rocket}_{altitude_km}_brightness"
        brightness = data[key]
        valid = np.isfinite(brightness)
        time_since_tg_s, y = resample_series_to_common_time(
            all_time_since_tg_s[valid],
            brightness[valid],
        )
        magnetic_lat_deg = interpolate_maglat(time_since_tg_s, rocket)
        traces.append(
            {
                "time_since_tg_s": time_since_tg_s,
                "magnetic_lat_deg": magnetic_lat_deg,
                "y": y,
                "name": f"{rocket} brightness",
                "color": plotly_color(ROCKET_COLORS[rocket]),
                "dash": "solid",
                "secondary_y": False,
                "type": "scatter",
                "render_mode": "svg",
            }
        )

    return {
        "label": f"Footpoint brightness {altitude_km} km",
        "left_y_title": "Brightness",
        "right_y_title": None,
        "left_y_type": "log",
        "source_files": source_file_names(metadata["source_data_file"]),
        "traces": traces,
    }


def load_erau_panel() -> dict:
    data = np.load(ERAU_NPZ)
    metadata = json.loads(str(data["metadata_json"]))
    traces = []
    units = metadata["series"][0]["units"]

    for series in metadata["series"]:
        source_time_since_tg_s = data[series["time_key"]]
        time_since_tg_s, signal = resample_series_to_common_time(
            source_time_since_tg_s,
            data[series["value_key"]],
        )
        magnetic_lat_deg = np.interp(
            time_since_tg_s,
            source_time_since_tg_s,
            data[series["maglat_key"]],
            left=np.nan,
            right=np.nan,
        )
        rocket = series["label"][:3]
        traces.append(
            {
                "time_since_tg_s": time_since_tg_s,
                "magnetic_lat_deg": magnetic_lat_deg,
                "y": signal,
                "name": series["label"],
                "color": plotly_color(ROCKET_COLORS[rocket]),
                "dash": "solid",
                "secondary_y": False,
                "type": "scatter",
                "render_mode": "svg",
            }
        )

    return {
        "label": "ERAU PIP",
        "left_y_title": f"Current ({units})",
        "right_y_title": None,
        "left_y_type": "linear",
        "source_files": source_file_names(metadata["source_data_file"]),
        "traces": traces,
    }


def load_erpa_hi_panel() -> dict:
    data = np.load(ERPA_HI_NPZ)
    metadata = json.loads(str(data["metadata_json"]))
    traces = []
    units = metadata["series"][0]["units"]

    for series in metadata["series"]:
        time_since_tg_s, hi = resample_series_to_common_time(
            data[series["time_key"]],
            data[series["value_key"]],
        )
        magnetic_lat_deg = interpolate_maglat(time_since_tg_s, series["rocket"])
        traces.append(
            {
                "time_since_tg_s": time_since_tg_s,
                "magnetic_lat_deg": magnetic_lat_deg,
                "y": hi,
                "name": series["label"],
                "color": plotly_color(series["color"]),
                "dash": "solid",
                "secondary_y": False,
                "type": "scatter",
                "render_mode": "svg",
            }
        )

    return {
        "label": "ERPA hi",
        "left_y_title": f"hi ({units})",
        "right_y_title": None,
        "left_y_type": "linear",
        "source_files": source_file_names(metadata["source_data_file"]),
        "traces": traces,
    }


def load_erpa_temp_panel() -> dict:
    data = np.load(ERPA_TEMP_NPZ)
    metadata = json.loads(str(data["metadata_json"]))
    traces = []
    units = metadata["series"][0]["units"]

    for series in metadata["series"]:
        time_since_tg_s, temp = resample_series_to_common_time(
            data[series["time_key"]],
            data[series["value_key"]],
        )
        magnetic_lat_deg = interpolate_maglat(time_since_tg_s, series["rocket"])
        traces.append(
            {
                "time_since_tg_s": time_since_tg_s,
                "magnetic_lat_deg": magnetic_lat_deg,
                "y": temp,
                "name": series["label"],
                "color": plotly_color(series["color"]),
                "dash": "solid",
                "secondary_y": False,
                "type": "scatter",
                "render_mode": "svg",
            }
        )

    return {
        "label": "ERPA temp",
        "left_y_title": f"Temp ({units})",
        "right_y_title": None,
        "left_y_type": "linear",
        "source_files": source_file_names(metadata["source_data_file"]),
        "traces": traces,
    }


def load_chimps_panels() -> dict:
    data = np.load(CHIMPS_NPZ)
    metadata = json.loads(str(data["metadata_json"]))
    time_since_tg_s = data["time_since_TG_s"]
    log10_energy_eV = data["log10_energy_eV"]
    log10_counts = data["log10_counts"]
    total_counts_time_since_tg_s, total_counts = resample_series_to_common_time(
        time_since_tg_s,
        data["total_counts"],
    )
    heatmap_time_since_tg_s, heatmap_log10_counts = resample_matrix_to_common_time(
        time_since_tg_s,
        log10_counts,
    )
    heatmap_maglat_deg = interpolate_maglat(
        heatmap_time_since_tg_s,
        metadata["rocket"],
    )
    total_counts_maglat_deg = interpolate_maglat(
        total_counts_time_since_tg_s,
        metadata["rocket"],
    )
    valid_heatmap_maglat = np.isfinite(heatmap_maglat_deg)
    zmin, zmax = metadata["log10_counts_limits"]

    return {
        "chimps_spectrogram": {
            "label": "CHIMPS downgoing electrons",
            "left_y_title": "Log10(Energy in eV)",
            "right_y_title": None,
            "left_y_type": "linear",
            "left_y_range": [2.25, 4.16],
            "source_files": source_file_names(metadata["source_data_file"]),
            "traces": [
                {
                    "type": "heatmap",
                    "time_since_tg_s": heatmap_time_since_tg_s,
                    "magnetic_lat_deg": heatmap_maglat_deg[valid_heatmap_maglat],
                    "y": log10_energy_eV,
                    "z": heatmap_log10_counts,
                    "z_maglat": heatmap_log10_counts[:, valid_heatmap_maglat],
                    "name": "397 CHIMPS downgoing electrons",
                    "colorscale": "Jet",
                    "zmin": zmin,
                    "zmax": zmax,
                    "colorbar_title": "log counts",
                    "secondary_y": False,
                },
            ],
        },
        "chimps_total_counts": {
            "label": "CHIMPS total counts",
            "left_y_title": "Total Counts",
            "right_y_title": None,
            "left_y_type": "linear",
            "left_y_range": [0, 30000],
            "source_files": source_file_names(metadata["source_data_file"]),
            "traces": [
                {
                    "type": "scatter",
                    "time_since_tg_s": total_counts_time_since_tg_s,
                    "magnetic_lat_deg": total_counts_maglat_deg,
                    "y": total_counts,
                    "name": "397 CHIMPS total counts",
                    "color": plotly_color(ROCKET_COLORS["397"]),
                    "dash": "solid",
                    "secondary_y": False,
                    "render_mode": "svg",
                },
            ],
        },
    }


def load_pip_voff_panels() -> dict:
    data = np.load(PIP_VOFF_NPZ)
    metadata = json.loads(str(data["metadata_json"]))
    panels = {}

    for series in metadata["series"]:
        rocket = series["rocket"]
        panels[f"pip_voff_{rocket}"] = {
            "label": f"{rocket} PIP3 Voff",
            "left_y_title": f"Voff ({series['units']})",
            "right_y_title": None,
            "left_y_type": "linear",
            "left_y_range": list(metadata["y_limits_v"]),
            "source_files": source_file_names(series["source_data_file"]),
            "traces": [
                {
                    "type": "scatter",
                    "time_since_tg_s": data[series["time_key"]],
                    "magnetic_lat_deg": data[series["maglat_key"]],
                    "y": data[series["value_key"]],
                    "name": series["label"],
                    "color": plotly_color(ROCKET_COLORS[rocket]),
                    "dash": "solid",
                    "secondary_y": False,
                    "render_mode": "svg",
                    "mode": "markers",
                    "marker": {
                        "color": plotly_color(ROCKET_COLORS[rocket]),
                        "size": 3,
                        "opacity": 0.25,
                    },
                },
            ],
        }

    return panels


def load_exb_panels() -> dict:
    data = np.load(EXB_NPZ)
    metadata = json.loads(str(data["metadata_json"]))
    panels = {}
    component_labels = {
        "east": "ExB east",
        "north": "ExB north",
        "up": "ExB up",
    }

    for series in metadata["series"]:
        component = series["component"]
        panel_id = f"exb_{component}"
        panels.setdefault(
            panel_id,
            {
                "label": component_labels.get(component, f"ExB {component}"),
                "left_y_title": f"{component_labels.get(component, component)} ({series['units']})",
                "right_y_title": None,
                "left_y_type": "linear",
                "left_y_range": metadata["y_limits"][component],
                "source_files": source_file_names(metadata["source_data_file"]),
                "traces": [],
            },
        )
        panels[panel_id]["traces"].append(
            {
                "type": "scatter",
                "time_since_tg_s": data[series["time_key"]],
                "magnetic_lat_deg": data[series["maglat_key"]],
                "y": data[series["value_key"]],
                "name": series["label"],
                "color": plotly_color(series["color"]),
                "dash": "solid",
                "secondary_y": False,
                "render_mode": "svg",
            }
        )

    return panels


def load_keogram_panels() -> dict:
    data = np.load(KEOGRAM_NPZ)
    metadata = json.loads(str(data["metadata_json"]))
    time_since_tg_s = data["time_since_tg_s"]
    vmin, vmax = data["brightness_limits"]
    color = str(data["color"].item()).capitalize()
    panels = {}

    for tag in [str(tag) for tag in data["tags"]]:
        brightness = data[f"brightness_{tag}"]
        flight_time_s = data[f"flight_time_{tag}_s"]
        trajectory_line_s = data[f"trajectory_line_{tag}_s"]
        y_min, y_max = data[f"y_limits_{tag}_s"]

        log_brightness = np.log10(np.clip(brightness, max(float(vmin), 1e-6), None))
        heatmap_time_since_tg_s, heatmap_z = resample_matrix_to_common_time(
            time_since_tg_s,
            log_brightness,
        )
        heatmap_maglat_deg = interpolate_maglat(heatmap_time_since_tg_s, tag)
        valid_heatmap_maglat = np.isfinite(heatmap_maglat_deg)
        valid_line = (
            np.isfinite(trajectory_line_s)
            & (trajectory_line_s >= y_min)
            & (trajectory_line_s <= y_max)
        )
        trajectory_time_since_tg_s, trajectory_line_s = resample_series_to_common_time(
            time_since_tg_s[valid_line],
            trajectory_line_s[valid_line],
        )
        trajectory_maglat_deg = interpolate_maglat(trajectory_time_since_tg_s, tag)

        panels[f"keogram_{tag}"] = {
            "label": f"{tag} trajectory keogram",
            "left_y_title": "Flight time since launch (s)",
            "right_y_title": None,
            "left_y_type": "linear",
            "source_files": source_file_names(metadata["source_data_file"]),
            "traces": [
                {
                    "type": "heatmap",
                    "time_since_tg_s": heatmap_time_since_tg_s,
                    "magnetic_lat_deg": heatmap_maglat_deg[valid_heatmap_maglat],
                    "y": flight_time_s,
                    "z": heatmap_z,
                    "z_maglat": heatmap_z[:, valid_heatmap_maglat],
                    "name": f"{tag} {color} keogram",
                    "colorscale": "Greens",
                    "zmin": float(np.log10(max(float(vmin), 1e-6))),
                    "zmax": float(np.log10(float(vmax))),
                    "colorbar_title": f"log {color} intensity",
                    "secondary_y": False,
                },
                {
                    "type": "scatter",
                    "time_since_tg_s": trajectory_time_since_tg_s,
                    "magnetic_lat_deg": trajectory_maglat_deg,
                    "y": trajectory_line_s,
                    "name": f"{tag} trajectory",
                    "color": plotly_color(ROCKET_COLORS.get(tag, "#d62728")),
                    "dash": "solid",
                    "secondary_y": False,
                    "render_mode": "svg",
                },
            ],
        }

    return panels


def load_panels() -> dict:
    panels = {}
    panels["erau"] = load_erau_panel()
    panels["erpa_hi"] = load_erpa_hi_panel()
    panels["erpa_temp"] = load_erpa_temp_panel()
    panels.update(load_chimps_panels())
    panels.update(load_pip_voff_panels())
    panels.update(load_exb_panels())
    panels.update(load_b_e_panels())
    panels["footpoint_brightness"] = load_footpoint_brightness_panel(altitude_km=110)
    panels.update(load_keogram_panels())
    return panels


PANEL_DEFS = load_panels()
PANEL_ORDER = [
    "chimps_spectrogram",
    "chimps_total_counts",
    "pip_voff_397",
    "pip_voff_398",
    "erau",
    "erpa_temp",
    "erpa_hi",
    "exb_east",
    "exb_north",
    "exb_up",
    "b_north_e_east",
    "b_east_e_north",
    "footpoint_brightness",
    "keogram_398",
    "keogram_397",
]
DEFAULT_PANELS = [panel_id for panel_id in PANEL_ORDER if panel_id in PANEL_DEFS]


def trace_x_values(trace: dict, use_maglat: bool):
    if use_maglat and "magnetic_lat_deg" in trace:
        return trace["magnetic_lat_deg"]
    if "time_since_tg_s" in trace:
        return trace["time_since_tg_s"]
    return trace["x"]


def heatmap_z_values(trace: dict, use_maglat: bool):
    if use_maglat and "z_maglat" in trace:
        return trace["z_maglat"]
    return trace["z"]


def stackplot_height(row_count: int) -> int:
    """Return a figure height that reserves a fixed gap between panel rows."""
    return max(360, PANEL_HEIGHT_PX * row_count + PANEL_GAP_PX * (row_count - 1))


def subplot_vertical_spacing(row_count: int) -> float:
    """Convert the fixed inter-panel pixel gap to Plotly domain units."""
    if row_count <= 1:
        return 0.0

    plot_height_px = stackplot_height(row_count) - 2 * PLOT_VERTICAL_MARGIN_PX
    return PANEL_GAP_PX / plot_height_px


def build_stackplot(
    selected_panel_ids: list[str],
    x_axis_mode: str = "time_since_TG",
    show_source_subtitles: bool = False,
) -> go.Figure:
    selected_panel_ids = [
        panel_id for panel_id in PANEL_ORDER if panel_id in selected_panel_ids
    ]
    use_maglat = x_axis_mode == "maglat"

    if not selected_panel_ids:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_white",
            height=300,
            annotations=[
                {
                    "text": "Select at least one panel.",
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.5,
                    "y": 0.5,
                    "showarrow": False,
                }
            ],
        )
        return fig

    specs = [[{"secondary_y": True}] for _ in selected_panel_ids]
    fig = make_subplots(
        rows=len(selected_panel_ids),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=subplot_vertical_spacing(len(selected_panel_ids)),
        specs=specs,
    )

    for row, panel_id in enumerate(selected_panel_ids, start=1):
        panel = PANEL_DEFS[panel_id]
        legend_items = []

        for trace in panel["traces"]:
            if trace["type"] == "heatmap":
                fig.add_trace(
                    go.Heatmap(
                        x=browser_values(trace_x_values(trace, use_maglat)),
                        y=browser_values(trace["y"]),
                        z=browser_values(heatmap_z_values(trace, use_maglat)),
                        name=trace["name"],
                        colorscale=trace["colorscale"],
                        zmin=trace["zmin"],
                        zmax=trace["zmax"],
                        colorbar=subplot_colorbar(
                            row,
                            len(selected_panel_ids),
                            trace["colorbar_title"],
                        ),
                        showscale=True,
                        hovertemplate=(
                            "Magnetic latitude: %{x:.3f} deg<extra></extra>"
                            if use_maglat
                            else "Time since TG: %{x:.2f} s<extra></extra>"
                        ),
                    ),
                    row=row,
                    col=1,
                    secondary_y=trace["secondary_y"],
                )
                continue

            legend_items.append(
                {
                    "name": trace["name"],
                    "color": trace["color"],
                    "dash": trace["dash"],
                }
            )
            is_keogram_trajectory = panel_id.startswith("keogram_")
            scatter_class = go.Scatter if trace.get("render_mode") == "svg" else go.Scattergl
            fig.add_trace(
                scatter_class(
                    x=browser_values(trace_x_values(trace, use_maglat)),
                    y=browser_values(trace["y"]),
                    mode=trace.get("mode", "lines"),
                    name=trace["name"],
                    line={
                        "color": trace["color"],
                        "dash": trace["dash"],
                        "width": 1.2,
                    },
                    marker=trace.get("marker"),
                    legendgroup=trace["name"],
                    showlegend=False,
                    hoverinfo="skip" if is_keogram_trajectory else "all",
                    hovertemplate=(
                        None
                        if is_keogram_trajectory
                        else f"{trace['name']}: %{{y:.6g}}<extra></extra>"
                    ),
                ),
                row=row,
                col=1,
                secondary_y=trace["secondary_y"],
            )

        fig.update_yaxes(
            title_text=panel["left_y_title"],
            title_font={"size": 15},
            tickfont={"size": 15},
            type=panel.get("left_y_type", "linear"),
            range=panel.get("left_y_range"),
            row=row,
            col=1,
            secondary_y=False,
        )
        if panel["right_y_title"]:
            fig.update_yaxes(
                title_text=panel["right_y_title"],
                title_font={"size": 15},
                tickfont={"size": 15},
                range=panel.get("right_y_range"),
                row=row,
                col=1,
                secondary_y=True,
            )
        else:
            fig.update_yaxes(showticklabels=False, row=row, col=1, secondary_y=True)
        add_panel_legend(fig, row, legend_items)
        add_panel_title(fig, row, panel_title(panel, show_source_subtitles))

    for row in range(1, len(selected_panel_ids) + 1):
        is_bottom_row = row == len(selected_panel_ids)
        x_axis_title = (
            "Magnetic latitude (deg)" if use_maglat else "Time since TG (s)"
        )
        fig.update_xaxes(
            range=None if use_maglat else list(TG_X_LIMITS_S),
            title_text=x_axis_title if is_bottom_row else None,
            title_font={"size": 15},
            tickfont={"size": 15},
            showticklabels=True,
            row=row,
            col=1,
        )
        fig.update_xaxes(
            unifiedhovertitle_text=(
                "Magnetic latitude: %{x:.3f} deg"
                if use_maglat
                else "Time since TG: %{x:.2f} s"
            ),
            row=row,
            col=1,
        )
    fig.update_layout(
        template="plotly_white",
        height=stackplot_height(len(selected_panel_ids)),
        margin={
            "l": 80,
            "r": 240,
            "t": PLOT_VERTICAL_MARGIN_PX,
            "b": PLOT_VERTICAL_MARGIN_PX,
        },
        hovermode="x unified",
        showlegend=False,
    )

    return fig


app = Dash(__name__)
app.layout = html.Div(
    [
        html.Div(
            [
                html.H2("GNEISS stackplot"),
                dcc.Checklist(
                    id="panel-selector",
                    options=[
                        {"label": PANEL_DEFS[panel_id]["label"], "value": panel_id}
                        for panel_id in PANEL_ORDER
                        if panel_id in PANEL_DEFS
                    ],
                    value=DEFAULT_PANELS,
                    inline=True,
                    inputStyle={"marginRight": "6px", "marginLeft": "14px"},
                ),
                html.Div(
                    [
                        html.Span("X-axis:", style={"fontWeight": "bold"}),
                        dcc.RadioItems(
                            id="x-axis-mode",
                            options=[
                                {
                                    "label": "Time since TG",
                                    "value": "time_since_TG",
                                },
                                {
                                    "label": "Magnetic latitude",
                                    "value": "maglat",
                                },
                            ],
                            value="time_since_TG",
                            inline=True,
                            inputStyle={"marginRight": "6px", "marginLeft": "14px"},
                        ),
                    ],
                    style={"display": "flex", "alignItems": "center", "marginTop": "8px"},
                ),
                html.Div(
                    [
                        dcc.Checklist(
                            id="source-subtitle-toggle",
                            options=[
                                {
                                    "label": "Show source data subtitles",
                                    "value": "show",
                                },
                            ],
                            value=[],
                            inline=True,
                            inputStyle={"marginRight": "6px"},
                        ),
                    ],
                    style={"marginTop": "8px"},
                ),
            ],
            style={
                "position": "sticky",
                "top": 0,
                "zIndex": 10,
                "background": "white",
                "borderBottom": "1px solid #ddd",
                "padding": "10px 16px",
            },
        ),
        dcc.Graph(
            id="stackplot",
            figure=build_stackplot(DEFAULT_PANELS, "time_since_TG"),
            config={"scrollZoom": True, "displaylogo": False},
            style={"width": "100%"},
        ),
    ],
    style={"fontFamily": "Arial, sans-serif"},
)


@app.callback(
    Output("stackplot", "figure"),
    Input("panel-selector", "value"),
    Input("x-axis-mode", "value"),
    Input("source-subtitle-toggle", "value"),
    prevent_initial_call=True,
)
def update_stackplot(selected_panel_ids, x_axis_mode, source_subtitle_toggle):
    show_source_subtitles = "show" in (source_subtitle_toggle or [])
    return build_stackplot(
        selected_panel_ids or [],
        x_axis_mode,
        show_source_subtitles,
    )


if __name__ == "__main__":
    app.run(debug=True, port=8051)
