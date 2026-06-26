# GNEISS Stackplot

This workspace contains scripts and prepared data products for an interactive
stackplot of GNEISS rocket measurements. The Dash app aligns the panels on a
shared x-axis and lets the viewer show or hide panels and switch between time
since TG and magnetic latitude.

## Workspace Layout

```text
GNEISS_stackplot/
├── data/
│   ├── app_data/
│   │   ├── b_e_field_components_data.npz
│   │   ├── chimps_397_downgoing_data.npz
│   │   ├── erau_signal_data.npz
│   │   ├── erpa_hi_data.npz
│   │   ├── erpa_temp_data.npz
│   │   ├── exb_components_data.npz
│   │   ├── footpoint_brightness_data.npz
│   │   ├── pip3_0_voff_data.npz
│   │   ├── tg_to_maglat.csv
│   │   └── trajectory_keogram_green_20260210_101900_102848.npz
│   └── source_data/
└── scripts/
    ├── build_app_data/
    │   ├── b_e_field.py
    │   ├── erau.py
    │   ├── erpa_hi.py
    │   ├── erpa_temp.py
    │   └── footpoint_brightness.py
    ├── stackplot_app.py
    └── requirements.txt
```

The plotting scripts create the NPZ data products. The instructions below
assume those NPZ files already exist and are ready to use.

## Run the Stackplot App

From the workspace root:

```bash
cd scripts
python3 -m pip install -r requirements.txt
python3 stackplot_app.py
```

Open [http://127.0.0.1:8051](http://127.0.0.1:8051) in a browser.

The app includes:

- ERAU PIP
- ERPA temperature
- ERPA hi
- CHIMPS downgoing electron energy-time spectrogram
- CHIMPS downgoing electron total counts
- PIP3 low-gain Voff for rockets 397 and 398
- ExB east, north, and up
- B north and E east
- B east and E north
- Footpoint brightness at 110 km
- Trajectory keograms for rockets 398 and 397

Use the checkboxes to show or hide panels. Use the x-axis control to switch all
panels between time since TG and magnetic latitude.

## Required Data

`scripts/stackplot_app.py` expects the prepared NPZ files and
`tg_to_maglat.csv` listed above under `data/app_data/`.

`data/source_data/` contains raw and intermediate inputs used to regenerate the
prepared NPZ files. The stackplot app does not read that folder directly.

## Data Processing Notes

The stackplot is not a raw-sample viewer. The prepared NPZ files and the app
apply a few display-oriented transformations:

- B/E field, ERPA hi, ERPA temperature, and ERAU PIP line series are capped at a
  maximum `0.05 s` TG-time resolution when the NPZ files are exported. If the
  source cadence is finer than that, values are linearly interpolated onto a
  `0.05 s` grid; coarser source series are left at their native finite sample
  times.
- ERAU PIP is additionally sorted by TG time, duplicate times are reduced to the
  first sample, the configured TG offset is applied, and a centered 100-sample
  rolling median is applied before export.
- CHIMPS 397 downgoing-electron data are sorted by time, duplicate times are
  reduced to the first sample, and `log10_counts` is stored as
  `log10(max(counts, 1))` for the energy-time spectrogram. The total-count line
  is stored without smoothing.
- PIP3 low-gain Voff uses the positive time samples from the sigmoid-fit
  exports to match the source notebook, and stores one separate series each for
  rockets 397 and 398 without smoothing or downsampling.
- In the Dash app, line panels are linearly interpolated onto the shared
  `0.05 s` TG grid for aligned display. Trajectory keogram image rows are
  linearly interpolated onto a shared `0.3 s` display grid. The CHIMPS
  spectrogram is also interpolated onto that `0.3 s` display grid.
- Magnetic latitude is not an independent source measurement in the NPZ series;
  it is linearly interpolated from `data/app_data/tg_to_maglat.csv` for the
  selected rocket and TG time samples.
- Keogram brightness is displayed as `log10` brightness with clipping at the
  stored brightness limits. Footpoint brightness values are read from the
  prepared brightness CSV and are only interpolated by the app onto the shared
  line-panel grid for display.
