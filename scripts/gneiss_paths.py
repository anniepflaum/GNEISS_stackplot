"""Portable data roots for GNEISS stackplot scripts."""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_DATA_DIR = PROJECT_ROOT / "data"


def source_data_dir() -> Path:
    if explicit := os.environ.get("GNEISS_SOURCE_DATA_DIR"):
        return Path(explicit).expanduser()
    if root := os.environ.get("LAB317_DATA_ROOT"):
        return Path(root).expanduser() / "raw" / "rocket" / "gneiss-stackplot"
    return LOCAL_DATA_DIR / "source_data"


def app_data_dir() -> Path:
    if explicit := os.environ.get("GNEISS_APP_DATA_DIR"):
        return Path(explicit).expanduser()
    if root := os.environ.get("LAB317_DATA_ROOT"):
        return Path(root).expanduser() / "processed" / "gneiss" / "stackplot"
    return LOCAL_DATA_DIR / "app_data"


def trajectory_data_dir() -> Path:
    if explicit := os.environ.get("ASI_TRAJECTORY_ROOT"):
        return Path(explicit).expanduser() / "GNEISS"
    lab_data_root = Path(
        os.environ.get("LAB317_DATA_ROOT", PROJECT_ROOT.parent / "data")
    ).expanduser()
    return lab_data_root / "raw" / "rocket" / "trajectories" / "GNEISS"
