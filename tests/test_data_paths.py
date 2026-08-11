from pathlib import Path
import sys


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from gneiss_paths import app_data_dir, source_data_dir  # noqa: E402


def test_lab317_data_root(monkeypatch, tmp_path):
    monkeypatch.delenv("GNEISS_SOURCE_DATA_DIR", raising=False)
    monkeypatch.delenv("GNEISS_APP_DATA_DIR", raising=False)
    monkeypatch.setenv("LAB317_DATA_ROOT", str(tmp_path))

    assert source_data_dir() == tmp_path / "raw" / "rocket" / "gneiss-stackplot"
    assert app_data_dir() == tmp_path / "processed" / "gneiss" / "stackplot"


def test_direct_overrides_take_precedence(monkeypatch, tmp_path):
    source = tmp_path / "source"
    app = tmp_path / "app"
    monkeypatch.setenv("LAB317_DATA_ROOT", str(tmp_path / "lab"))
    monkeypatch.setenv("GNEISS_SOURCE_DATA_DIR", str(source))
    monkeypatch.setenv("GNEISS_APP_DATA_DIR", str(app))

    assert source_data_dir() == source
    assert app_data_dir() == app
