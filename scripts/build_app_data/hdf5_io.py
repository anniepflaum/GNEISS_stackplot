from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np


def write_hdf5(
    output_path: str | Path,
    arrays: dict[str, np.ndarray],
    **json_attributes,
) -> Path:
    """Write compressed numerical datasets and JSON-encoded file attributes."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as h5:
        h5.attrs["format_version"] = 1
        for name, value in json_attributes.items():
            h5.attrs[name] = json.dumps(value)

        for name, values in arrays.items():
            data = np.asarray(values)
            create_options = {}
            if data.ndim > 0:
                create_options = {
                    "compression": "gzip",
                    "compression_opts": 4,
                    "shuffle": data.dtype.kind not in {"U", "S", "O"},
                    "chunks": True,
                }
            if data.dtype.kind == "U":
                data = data.astype(object)
                create_options["dtype"] = h5py.string_dtype(encoding="utf-8")
            h5.create_dataset(name, data=data, **create_options)

    return output_path


def read_hdf5(output_path: str | Path) -> dict:
    """Load all datasets and JSON attributes into an in-memory mapping."""
    with h5py.File(output_path, "r") as h5:
        data = {}
        for name, dataset in h5.items():
            values = dataset.asstr()[...] if h5py.check_string_dtype(dataset.dtype) else dataset[...]
            if isinstance(values, np.ndarray) and values.ndim == 0:
                values = values.item()
            data[name] = values
        data.update({name: value for name, value in h5.attrs.items()})
    return data
