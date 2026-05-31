from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
BRIGHTNESS_CSV = DATA_DIR / "traj_brightness" / "brightness_vs_time_20260210_101900_102848_step0p05.csv"
FOOTPOINT_BRIGHTNESS_NPZ = DATA_DIR / "footpoint_brightness_data.npz"
TRAJECTORY_FILES = {
    "397": Path(
        "/Users/anniepflaum/asi_mapping/trajectories/GNEISS/"
        "36397_GPS_Time_Export_01.csv"
    ),
    "398": Path(
        "/Users/anniepflaum/asi_mapping/trajectories/GNEISS/"
        "36398_GPS_Time_Export_00.csv"
    ),
}
ASI_IMAGE_SOURCE = "ARV/VEE/BVR GASI_5577 TIFFs from optics.gi.alaska.edu/amisr_archive"
ALTITUDES_KM = (95, 100, 105, 110)


def export_footpoint_brightness_npz(
    csv_path: str | Path = BRIGHTNESS_CSV,
    output_path: str | Path = FOOTPOINT_BRIGHTNESS_NPZ,
):
    """Export stackplot-ready brightness arrays with trajectory provenance."""
    csv_path = Path(csv_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(csv_path)
    arrays = {
        "time_since_TG_s": data["TG"].to_numpy(),
    }

    for rocket in ("397", "398"):
        for altitude_km in ALTITUDES_KM:
            column = f"{rocket}_{altitude_km}_brightness"
            arrays[column] = data[column].to_numpy()

    metadata = {
        "source_data_file": {
            **{rocket: path.name for rocket, path in TRAJECTORY_FILES.items()},
            "asi_images": ASI_IMAGE_SOURCE,
        },
        "source_brightness_csv": csv_path.name,
        "time_key": "time_since_TG_s",
        "brightness_key_template": "{rocket}_{altitude_km}_brightness",
    }

    np.savez_compressed(
        output_path,
        **arrays,
        metadata_json=np.array(json.dumps(metadata)),
    )

    return output_path


if __name__ == "__main__":
    export_footpoint_brightness_npz()
