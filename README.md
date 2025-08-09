# Heating & Cooling Load Disaggregation Pipeline

**Purpose:**  
Disaggregate building energy meter data into heating/cooling components using XGBoost, comparing a baseline model (time/lag features) and an augmented model (with FFT/spectral features).

## Structure

- `data_utils.py` – Data loading, datetime handling, optimization
- `feature_engineering.py` – Feature extraction: time, rolling, lag, FFT
- `models.py` – XGBoost classifier/regressor wrappers
- `evaluation.py` – Metrics, CSV aggregation
- `plot_utils.py` – Time series & spectral plots
- `process.py` – Batch processing test files, report generation
- `main.py` – CLI entrypoint, config setup

## Usage

1. Place your data in `data/` (see `main.py` for expected input filenames).
2. Install dependencies:  
