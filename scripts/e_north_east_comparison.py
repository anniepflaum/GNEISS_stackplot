from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
SOURCE_DATA_DIR = DATA_DIR / "source_data"

E_FIELD_CSVS = {
    "397": SOURCE_DATA_DIR / "e_despun_subtracted_v1_397.csv",
    "398": SOURCE_DATA_DIR / "e_despun_subtracted_v1_398.csv",
}
FLOW_CSVS = {
    "397": SOURCE_DATA_DIR / "exb_v1_397.csv",
    "398": SOURCE_DATA_DIR / "exb_v1_398.csv",
}
T0_TG_OFFSET_S = {
    "397": 0.3167,
    "398": 30.4242,
}
ROCKET_COLORS = {
    "397": "tab:blue",
    "398": "tab:orange",
}
X_LIMITS_S = (100.0, 500.0)
ROBUST_Y_PERCENTILES = (0.1, 99.9)

PANELS = (
    {
        "title": "E north",
        "source": "e",
        "column": 2,
        "ylabel": "E north (V/m)",
    },
    {
        "title": "Flow east",
        "source": "flow",
        "column": 1,
        "ylabel": "Flow east (m/s)",
    },
    {
        "title": "E east",
        "source": "e",
        "column": 1,
        "ylabel": "E east (V/m)",
    },
    {
        "title": "Flow north",
        "source": "flow",
        "column": 2,
        "ylabel": "Flow north (m/s)",
    },
)


def load_component(csv_path: str | Path, component_column: int, rocket: str):
    """Load one component and convert local file time to time since TG."""
    data = np.loadtxt(Path(csv_path), delimiter=",", comments="#")
    time_since_tg_s = data[:, 0] + T0_TG_OFFSET_S[rocket]
    values = data[:, component_column]
    return time_since_tg_s, values


def source_csvs(source: str):
    if source == "e":
        return E_FIELD_CSVS
    if source == "flow":
        return FLOW_CSVS
    raise ValueError(f"Unknown source: {source}")


def source_file_subtitle():
    source_files = [
        path.name
        for csvs in (E_FIELD_CSVS, FLOW_CSVS)
        for path in csvs.values()
    ]
    return "Source files: " + ", ".join(source_files)


def robust_y_limits(series):
    visible_values = []

    for time_since_tg_s, values in series:
        in_window = (
            np.isfinite(time_since_tg_s)
            & np.isfinite(values)
            & (time_since_tg_s >= X_LIMITS_S[0])
            & (time_since_tg_s <= X_LIMITS_S[1])
        )
        visible_values.append(values[in_window])

    values = np.concatenate(visible_values)
    if len(values) == 0:
        return None

    low, high = np.nanpercentile(values, ROBUST_Y_PERCENTILES)
    if not np.isfinite(low) or not np.isfinite(high):
        return None
    if low == high:
        margin = abs(low) * 0.05 if low else 1.0
        return low - margin, high + margin

    margin = 0.05 * (high - low)
    return low - margin, high + margin


def plot_e_flow_comparison():
    fig, axes = plt.subplots(len(PANELS), 1, figsize=(12, 9), sharex=True)
    fig.suptitle(source_file_subtitle(), fontsize=9, y=0.995)

    for ax, panel in zip(axes, PANELS):
        panel_series = []
        for rocket, csv_path in source_csvs(panel["source"]).items():
            time_since_tg_s, values = load_component(
                csv_path,
                component_column=panel["column"],
                rocket=rocket,
            )
            panel_series.append((time_since_tg_s, values))
            ax.plot(
                time_since_tg_s,
                values,
                color=ROCKET_COLORS[rocket],
                linewidth=0.8,
                label=rocket,
            )

        ax.set_title(panel["title"])
        ax.set_ylabel(panel["ylabel"])
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", framealpha=0.95)
        y_limits = robust_y_limits(panel_series)
        if y_limits is not None:
            ax.set_ylim(*y_limits)

    axes[-1].set_xlabel("Time since TG (s)")
    axes[-1].set_xlim(*X_LIMITS_S)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show()
    return fig, axes


if __name__ == "__main__":
    plot_e_flow_comparison()
