from __future__ import annotations

import numpy as np


TIME_STEP_S = 0.05


def reduce_series_resolution(time_since_tg_s, values, time_step_s: float = TIME_STEP_S):
    """Cap a finite line series at a TG-aligned fixed-time resolution."""
    time_since_tg_s = np.asarray(time_since_tg_s)
    values = np.asarray(values)
    valid = np.isfinite(time_since_tg_s) & np.isfinite(values)
    source_time_s = time_since_tg_s[valid]
    source_values = values[valid]

    if len(source_time_s) == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    positive_steps = np.diff(source_time_s)
    positive_steps = positive_steps[positive_steps > 0]
    if len(positive_steps) == 0 or np.median(positive_steps) >= time_step_s:
        return source_time_s, source_values

    first_sample = int(np.ceil(np.min(source_time_s) / time_step_s))
    last_sample = int(np.floor(np.max(source_time_s) / time_step_s))
    target_time_s = np.round(
        np.arange(first_sample, last_sample + 1) * time_step_s,
        decimals=10,
    )
    return target_time_s, np.interp(target_time_s, source_time_s, source_values)
