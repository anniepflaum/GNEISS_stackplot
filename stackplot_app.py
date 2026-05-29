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
B_E_NPZ = DATA_DIR / "b_e_field_components_data.npz"
KEOGRAM_NPZ = DATA_DIR / "trajectory_keogram_green_20260210_101900_102848.npz"
BRIGHTNESS_CSV = (
    DATA_DIR
    / "brightness_vs_time_20260210_101900_102848_step0p05.csv"
)
TG_X_LIMITS_S = (0.0, 588.0)
PANEL_HEIGHT_PX = 260
MAX_POINTS_PER_TRACE = 60_000
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


def decimate_for_display(x, y, max_points: int = MAX_POINTS_PER_TRACE):
    if len(x) <= max_points:
        return x, y

    stride = int(np.ceil(len(x) / max_points))
    return x[::stride], y[::stride]


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
            font={"size": 12},
            bgcolor="rgba(255,255,255,0.85)",
        )


def load_b_e_panels() -> dict:
    npz = np.load(B_E_NPZ)
    metadata = json.loads(str(npz["metadata_json"]))
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
                "traces": [],
            },
        )

        use_right_axis = series["source"] == "e"
        if use_right_axis:
            panels[panel_id]["right_y_title"] = f"{series['component']} ({series['units']})"
        else:
            panels[panel_id]["left_y_title"] = f"{series['component']} ({series['units']})"

        x, y = decimate_for_display(npz[series["time_key"]], npz[series["value_key"]])
        panels[panel_id]["traces"].append(
            {
                "x": x,
                "y": y,
                "name": f"{series['rocket']} {series['component']}",
                "color": plotly_color(series["color"]),
                "dash": "dash" if series["line_style"] == "dashed" else "solid",
                "secondary_y": use_right_axis,
                "type": "scatter",
            }
        )

    return panels


def load_footprint_brightness_panel(altitude_km: int = 110) -> dict:
    data = pd.read_csv(BRIGHTNESS_CSV)
    traces = []

    for rocket in ("397", "398"):
        column = f"{rocket}_{altitude_km}_brightness"
        valid = data[column].notna()
        x, y = decimate_for_display(
            data.loc[valid, "TG"].to_numpy(),
            data.loc[valid, column].to_numpy(),
        )
        traces.append(
            {
                "x": x,
                "y": y,
                "name": f"{rocket} brightness",
                "color": plotly_color(ROCKET_COLORS[rocket]),
                "dash": "solid",
                "secondary_y": False,
                "type": "scatter",
            }
        )

    return {
        "label": f"Footprint brightness {altitude_km} km",
        "left_y_title": "Brightness",
        "right_y_title": None,
        "left_y_type": "log",
        "traces": traces,
    }


def load_keogram_panels() -> dict:
    data = np.load(KEOGRAM_NPZ)
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
        valid_line = (
            np.isfinite(trajectory_line_s)
            & (trajectory_line_s >= y_min)
            & (trajectory_line_s <= y_max)
        )

        panels[f"keogram_{tag}"] = {
            "label": f"{tag} trajectory keogram",
            "left_y_title": "Flight time since launch (s)",
            "right_y_title": None,
            "left_y_type": "linear",
            "traces": [
                {
                    "type": "heatmap",
                    "x": time_since_tg_s,
                    "y": flight_time_s,
                    "z": log_brightness,
                    "name": f"{tag} {color} keogram",
                    "colorscale": "Greens",
                    "zmin": float(np.log10(max(float(vmin), 1e-6))),
                    "zmax": float(np.log10(float(vmax))),
                    "colorbar_title": f"log {color} intensity",
                    "secondary_y": False,
                },
                {
                    "type": "scatter",
                    "x": time_since_tg_s[valid_line],
                    "y": trajectory_line_s[valid_line],
                    "name": f"{tag} trajectory",
                    "color": plotly_color(ROCKET_COLORS.get(tag, "#d62728")),
                    "dash": "solid",
                    "secondary_y": False,
                },
            ],
        }

    return panels


def load_panels() -> dict:
    panels = {}
    panels.update(load_b_e_panels())
    panels["footprint_brightness"] = load_footprint_brightness_panel(altitude_km=110)
    panels.update(load_keogram_panels())
    return panels


PANEL_DEFS = load_panels()
DEFAULT_PANELS = list(PANEL_DEFS)


def build_stackplot(selected_panel_ids: list[str]) -> go.Figure:
    selected_panel_ids = [panel_id for panel_id in selected_panel_ids if panel_id in PANEL_DEFS]

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
    subplot_titles = [PANEL_DEFS[panel_id]["label"] for panel_id in selected_panel_ids]
    fig = make_subplots(
        rows=len(selected_panel_ids),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.035,
        specs=specs,
        subplot_titles=subplot_titles,
    )

    for row, panel_id in enumerate(selected_panel_ids, start=1):
        panel = PANEL_DEFS[panel_id]
        legend_items = []

        for trace in panel["traces"]:
            if trace["type"] == "heatmap":
                fig.add_trace(
                    go.Heatmap(
                        x=trace["x"],
                        y=trace["y"],
                        z=trace["z"],
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
                            "TG: %{x:.2f} s<br>"
                            "log intensity: %{z:.3f}<extra></extra>"
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
            fig.add_trace(
                go.Scattergl(
                    x=trace["x"],
                    y=trace["y"],
                    mode="lines",
                    name=trace["name"],
                    line={
                        "color": trace["color"],
                        "dash": trace["dash"],
                        "width": 1.2,
                    },
                    legendgroup=trace["name"],
                    showlegend=False,
                ),
                row=row,
                col=1,
                secondary_y=trace["secondary_y"],
            )

        fig.update_yaxes(
            title_text=panel["left_y_title"],
            type=panel.get("left_y_type", "linear"),
            row=row,
            col=1,
            secondary_y=False,
        )
        if panel["right_y_title"]:
            fig.update_yaxes(title_text=panel["right_y_title"], row=row, col=1, secondary_y=True)
        else:
            fig.update_yaxes(showticklabels=False, row=row, col=1, secondary_y=True)
        add_panel_legend(fig, row, legend_items)

    fig.update_xaxes(range=list(TG_X_LIMITS_S), title_text="Time since TG (s)", row=len(selected_panel_ids), col=1)
    fig.update_layout(
        template="plotly_white",
        height=max(360, PANEL_HEIGHT_PX * len(selected_panel_ids)),
        margin={"l": 80, "r": 240, "t": 55, "b": 55},
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
                        {"label": panel["label"], "value": panel_id}
                        for panel_id, panel in PANEL_DEFS.items()
                    ],
                    value=DEFAULT_PANELS,
                    inline=True,
                    inputStyle={"marginRight": "6px", "marginLeft": "14px"},
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
            config={"scrollZoom": True, "displaylogo": False},
            style={"width": "100%"},
        ),
    ],
    style={"fontFamily": "Arial, sans-serif"},
)


@app.callback(
    Output("stackplot", "figure"),
    Input("panel-selector", "value"),
)
def update_stackplot(selected_panel_ids):
    return build_stackplot(selected_panel_ids or [])


if __name__ == "__main__":
    app.run(debug=True)
