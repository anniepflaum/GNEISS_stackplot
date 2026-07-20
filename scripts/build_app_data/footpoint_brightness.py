from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

try:
    from .hdf5_io import write_hdf5
except ImportError:
    from hdf5_io import write_hdf5


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parents[1] / "data"
APP_DATA_DIR = DATA_DIR / "app_data"
SOURCE_DATA_DIR = DATA_DIR / "source_data"
BRIGHTNESS_H5 = SOURCE_DATA_DIR / "brightness_vs_time_20260210_101900_102848_step0p05.h5"
FOOTPOINT_BRIGHTNESS_H5 = APP_DATA_DIR / "footpoint_brightness_data.h5"
TRAJECTORY_FILES = {
    "397": Path(
        "/Users/anniepflaum/lab317/asi_mapping/trajectories/GNEISS/"
        "36397_GPS_Time_Export_01.csv"
    ),
    "398": Path(
        "/Users/anniepflaum/lab317/asi_mapping/trajectories/GNEISS/"
        "36398_GPS_Time_Export_00.csv"
    ),
}
ASI_IMAGE_SOURCE = "ARV/VEE/BVR GASI_5577 TIFFs from optics.gi.alaska.edu/amisr_archive"
ALTITUDES_KM = (110,)


def export_footpoint_brightness_hdf5(
    brightness_h5_path: str | Path = BRIGHTNESS_H5,
    output_path: str | Path = FOOTPOINT_BRIGHTNESS_H5,
):
    """Export stackplot-ready brightness arrays with trajectory provenance."""
    brightness_h5_path = Path(brightness_h5_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(brightness_h5_path, "r") as source:
        if source.attrs.get("brightness_units") != "Rayleighs":
            raise ValueError(f"{brightness_h5_path} is not Rayleigh-calibrated")
        arrays = {
            "time_since_TG_s": np.asarray(source["time_since_tg_s"], dtype=float),
        }
        for rocket in ("397", "398"):
            for altitude_km in ALTITUDES_KM:
                column = f"{rocket}_{altitude_km}_brightness"
                arrays[column] = np.asarray(
                    source[f"rockets/{rocket}/brightness/{altitude_km}_km"],
                    dtype=float,
                )
        calibration_json = str(source.attrs["calibration_json"])

    metadata = {
        "source_data_file": {
            **{rocket: path.name for rocket, path in TRAJECTORY_FILES.items()},
            "asi_images": ASI_IMAGE_SOURCE,
        },
        "source_brightness_h5": brightness_h5_path.name,
        "brightness_units": "Rayleighs",
        "calibration_json": calibration_json,
        "time_key": "time_since_TG_s",
        "brightness_key_template": "{rocket}_{altitude_km}_brightness",
    }

    return write_hdf5(
        output_path,
        arrays,
        metadata_json=metadata,
    )


if __name__ == "__main__":
    export_footpoint_brightness_hdf5()
