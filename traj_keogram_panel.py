from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
DEFAULT_NPZ = DATA_DIR / "traj_keo" / "trajectory_keogram_green_20260210_101900_102848.npz"
DEFAULT_OUTPUT = DATA_DIR / "traj_keo" / "trajectory_keogram_panel.png"

ROCKET_COLORS = {
    "397": "tab:blue",
    "398": "tab:orange",
}


def _scalar_text(data, key: str, default: str = "") -> str:
    if key not in data:
        return default
    value = data[key]
    if getattr(value, "shape", ()) == ():
        return str(value.item())
    return str(value)


def plot_traj_keogram(
    npz_path: str | Path = DEFAULT_NPZ,
    axes=None,
    output_path: str | Path | None = None,
    show: bool = True,
    add_colorbar: bool = True,
    set_xlabel: bool = True,
):
    """Plot the trajectory keogram from an exported asi_mapping NPZ file.

    Pass existing matplotlib axes when composing this panel into a larger
    stackplot. The axes length must match the number of rocket panels in the
    NPZ file.
    """
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Missing keogram NPZ: {npz_path}\n"
            "Generate it from asi_mapping with scripts/traj_keogram.py, then "
            "place it under data/traj_keo/."
        )

    data = np.load(npz_path, allow_pickle=False)
    tags = [str(tag) for tag in data["tags"]]
    if axes is None:
        fig_height = 3.0 * len(tags)
        fig, axes = plt.subplots(
            len(tags),
            1,
            figsize=(10, fig_height),
            sharex=True,
            constrained_layout=True,
        )
    else:
        axes = np.atleast_1d(axes).ravel()
        if len(axes) != len(tags):
            raise ValueError(f"Expected {len(tags)} axes, got {len(axes)}")
        fig = axes[0].figure

    axes = np.atleast_1d(axes).ravel()
    x_min, x_max = data["x_limits_s"]
    vmin, vmax = data["brightness_limits"]
    time_since_tg_s = data["time_since_tg_s"]
    color = _scalar_text(data, "color", "green")
    norm = mpl.colors.LogNorm(vmin=max(float(vmin), 1e-6), vmax=float(vmax))

    image_handle = None
    for ax, tag in zip(axes, tags):
        brightness = data[f"brightness_{tag}"]
        flight_time_s = data[f"flight_time_{tag}_s"]
        y_min, y_max = data[f"y_limits_{tag}_s"]
        line_y = data[f"trajectory_line_{tag}_s"]

        image_handle = ax.imshow(
            brightness,
            origin="lower",
            aspect="auto",
            extent=[float(x_min), float(x_max), float(y_min), float(y_max)],
            cmap="Greens",
            norm=norm,
        )
        valid_line = np.isfinite(line_y) & (line_y >= y_min) & (line_y <= y_max)
        if np.any(valid_line):
            ax.plot(
                time_since_tg_s[valid_line],
                line_y[valid_line],
                color=ROCKET_COLORS.get(tag, "#ff3b30"),
                linewidth=1.4,
                label=f"{tag} trajectory",
            )
            ax.legend(loc="upper right", framealpha=0.95)

        ax.set_title(f"{tag} Trajectory Keogram")
        ax.set_ylabel("Flight Time Since Launch (s)")
        ax.set_ylim(float(y_min), float(y_max))
        ax.grid(False)

    if set_xlabel:
        axes[-1].set_xlabel("Time since TG (s)")

    if add_colorbar and image_handle is not None:
        cbar = fig.colorbar(image_handle, ax=axes, orientation="vertical", shrink=0.95)
        cbar.set_label(f"{color.capitalize()} Channel Intensity")

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)

    if show:
        plt.show()

    return fig, axes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", default=str(DEFAULT_NPZ), help="Exported keogram NPZ from asi_mapping")
    parser.add_argument("--output", default=None, help="Optional output PNG path")
    parser.add_argument("--no-show", action="store_true", help="Save or build the figure without opening a window")
    parser.add_argument("--no-colorbar", action="store_true", help="Do not add a keogram colorbar")
    args = parser.parse_args()

    output_path = args.output
    if output_path is None and args.no_show:
        output_path = DEFAULT_OUTPUT

    plot_traj_keogram(
        npz_path=args.npz,
        output_path=output_path,
        show=not args.no_show,
        add_colorbar=not args.no_colorbar,
    )


if __name__ == "__main__":
    main()
