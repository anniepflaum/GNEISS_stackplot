from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
SOURCE_DATA_DIR = DATA_DIR / "source_data"

DESPIN_CSVS = {
    "397": SOURCE_DATA_DIR / "despin_v2_397.csv",
    "398": SOURCE_DATA_DIR / "despin_v2_398.csv",
}
SUBTRACTED_CSVS = {
    "397": SOURCE_DATA_DIR / "e_despun_subtracted_v1_397.csv",
    "398": SOURCE_DATA_DIR / "e_despun_subtracted_v1_398.csv",
}
T0_TG_OFFSET_S = {
    "397": 0.3167,
    "398": 30.4242,
}
TRACE_COLORS = {
    ("397", "despin_v2"): "#8ecae6",
    ("397", "subtracted"): "tab:blue",
    ("398", "despin_v2"): "#ffb36b",
    ("398", "subtracted"): "tab:orange",
}
X_LIMITS_S = (100.0, 500.0)

PANELS = (
    {
        "title": "E east",
        "column": 1,
        "ylabel": "E east (V/m)",
        "ylim": (-0.04, 0.06),
    },
    {
        "title": "E north",
        "column": 2,
        "ylabel": "E north (V/m)",
        "ylim": (-0.03, 0.035),
    },
)


def load_component(csv_path: str | Path, component_column: int, rocket: str):
    """Load one component and convert local file time to time since TG."""
    data = np.loadtxt(Path(csv_path), delimiter=",", comments="#")
    time_since_tg_s = data[:, 0] + T0_TG_OFFSET_S[rocket]
    values = data[:, component_column]
    return time_since_tg_s, values


def source_file_subtitle():
    source_files = [
        path.name
        for csvs in (DESPIN_CSVS, SUBTRACTED_CSVS)
        for path in csvs.values()
    ]
    return "Source files: " + ", ".join(source_files)


def plot_subtracted_v_despun():
    fig, axes = plt.subplots(len(PANELS), 1, figsize=(12, 7), sharex=True)
    fig.suptitle(source_file_subtitle(), fontsize=9, y=0.995)

    source_groups = (
        ("despin_v2", DESPIN_CSVS),
        ("subtracted", SUBTRACTED_CSVS),
    )

    for ax, panel in zip(axes, PANELS):
        for file_type, csvs in source_groups:
            for rocket, csv_path in csvs.items():
                time_since_tg_s, values = load_component(
                    csv_path,
                    component_column=panel["column"],
                    rocket=rocket,
                )
                ax.plot(
                    time_since_tg_s,
                    values,
                    color=TRACE_COLORS[(rocket, file_type)],
                    linewidth=0.8,
                    label=f"{rocket} {file_type}",
                )

        ax.set_title(panel["title"])
        ax.set_ylabel(panel["ylabel"])
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", framealpha=0.95)
        ax.set_ylim(*panel["ylim"])

    axes[-1].set_xlabel("Time since TG (s)")
    axes[-1].set_xlim(*X_LIMITS_S)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    plt.show()
    return fig, axes


if __name__ == "__main__":
    plot_subtracted_v_despun()
