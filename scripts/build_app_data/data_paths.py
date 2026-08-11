"""Make the parent scripts directory's data-root helpers importable here."""

import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from gneiss_paths import (  # noqa: E402,F401
    app_data_dir,
    source_data_dir,
    trajectory_data_dir,
)
