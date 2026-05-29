from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from b_e_field import TG_X_LIMITS_S


SCRIPT_DIR = Path(__file__).resolve().parent
BRIGHTNESS_CSV = (
    SCRIPT_DIR.parent
    / "data"
    / "traj_brightness"
    / "brightness_vs_time_20260210_101900_102848_step0p05.csv"
)
ROCKET_COLORS = {
    "397": "tab:blue",
    "398": "tab:orange",
}


def load_footprint_brightness(
    csv_path: str | Path = BRIGHTNESS_CSV,
    rocket: str = "397",
    altitude_km: int = 110,
):
    """Load footprint brightness versus TG for one rocket and altitude."""
    time_s = []
    brightness = []
    brightness_column = f"{rocket}_{altitude_km}_brightness"

    with Path(csv_path).open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            value = row[brightness_column]
            if not value:
                continue

            time_s.append(float(row["TG"]))
            brightness.append(float(value))

    return np.array(time_s), np.array(brightness)


def plot_footprint_brightness(
    csv_path: str | Path = BRIGHTNESS_CSV,
    altitude_km: int = 110,
    output_path: str | Path | None = None,
    show: bool = True,
):
    """Plot footprint brightness for rockets 397 and 398."""
    fig, ax = plt.subplots(figsize=(10, 4))

    for rocket in ("397", "398"):
        time_s, brightness = load_footprint_brightness(
            csv_path=csv_path,
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

    ax.set_title(f"Footprint brightness")
    ax.set_xlabel("Time since TG (s)")
    ax.set_ylabel("Brightness")
    ax.set_xlim(*TG_X_LIMITS_S)
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


if __name__ == "__main__":
    plot_footprint_brightness(show=True)
