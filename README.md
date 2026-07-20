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
│   │   ├── b_e_field_components_data.h5
│   │   ├── chimps_397_downgoing_data.h5
│   │   ├── erau_signal_data.h5
│   │   ├── erpa_hi_data.h5
│   │   ├── erpa_temp_data.h5
│   │   ├── exb_components_data.h5
│   │   ├── footpoint_brightness_data.h5
│   │   ├── pip3_0_voff_data.h5
│   │   ├── tg_to_maglat.csv
│   │   └── trajectory_keogram_green_20260210_101900_102848.h5
│   └── source_data/
│       └── brightness_vs_time_20260210_101900_102848_step0p05.h5
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

The build scripts create full-resolution HDF5 data products. The instructions
below assume those HDF5 files already exist.

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
- (ExB)/B^2 east, north, and up (hidden as of 7/20/26)
- B north and E east
- B east and E north
- Footpoint brightness at 110 km
- Trajectory keograms for rockets 398 and 397

Use the checkboxes to show or hide panels. Use the x-axis control to switch all
panels between time since TG and magnetic latitude.

## Required Data

`scripts/stackplot_app.py` expects the prepared HDF5 files and
`tg_to_maglat.csv` listed above under `data/app_data/`.

`data/source_data/` contains raw and intermediate inputs used to regenerate the
prepared HDF5 files. The stackplot app does not read that folder directly.

## Data Processing Notes

The prepared HDF5 files retain the source resolution. The app applies a few
display-oriented transformations:

- B/E field, ExB, ERPA hi, ERPA temperature, and ERAU PIP series are written to
  HDF5 at their full processed source cadence without export-time downsampling.
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
- The Dash app does not interpolate panels onto a shared time grid. Lower-rate
  panels use their stored HDF5 timestamps directly. Full-view B/E and ERAU
  traces use an extrema-preserving display aggregate; zooming reads only the
  visible HDF5 interval and returns every native sample once the interval fits
  within the display point limit.
- Magnetic latitude is not an independent source measurement in the HDF5 series;
  it is linearly interpolated from `data/app_data/tg_to_maglat.csv` for the
  selected rocket and TG time samples.
- Keogram brightness is displayed as `log10` brightness with clipping at the
  stored brightness limits. Footpoint brightness retains the finite timestamps
  from the prepared brightness CSV.
