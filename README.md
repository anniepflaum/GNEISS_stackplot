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
│   │   ├── erau_signal_data.npz
│   │   ├── erpa_hi_data.npz
│   │   ├── erpa_temp_data.npz
│   │   ├── footpoint_brightness_data.npz
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
