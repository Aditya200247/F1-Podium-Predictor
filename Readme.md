# F1 Podium Predictor

A machine learning project that predicts Formula 1 podium finishers (top 3) using historical race data from 2000–2024. Built as an end-to-end practice project covering data cleaning, leakage-safe feature engineering, model training, and rigorous time-series validation.

## Overview

Given a race's grid positions, championship standings, and recent form, this project predicts which drivers are most likely to finish on the podium. It started as a simple winner-prediction exercise and evolved into a podium-prediction model after discovering that podium predictions are statistically more robust to evaluate (3x more positive examples than win-only prediction) while still being a meaningful target.

The project deliberately prioritizes methodological rigor over chasing a high accuracy number — most of the value here is in the validation approach (chronological splits, leakage prevention, walk-forward testing across multiple years) rather than the final metric itself.

## Dataset

This project uses the Ergast-schema Formula 1 historical dataset, commonly distributed on Kaggle (search "Formula 1 World Championship" or "Formula 1 Race Data"). It is not included in this repo — download it yourself and place the CSVs in a `data/` folder.

The dataset is a snapshot frozen at a point in time and does not update automatically; check the file modification dates after downloading to know which season it ends at. The original data is sourced from the Ergast Developer API (now retired) and its community successor, the Jolpica-F1 API.

Of the ~14 CSVs included, this project uses: `races.csv`, `results.csv`, `drivers.csv`, `constructors.csv`, `driver_standings.csv`, `constructor_standings.csv`, `qualifying.csv`, `circuits.csv`, and `status.csv`. The lap-by-lap and pit-stop files (`lap_times.csv`, `pit_stops.csv`) are not used.

## Project structure

```
.
├── data/                       # place downloaded CSVs here (not committed)
├── build_features.py           # builds the core feature table (model_table.csv)
├── build_features_v2.py        # extended feature table with qualifying gap, DNF rate, circuit history
├── train_baseline.py           # logistic regression + XGBoost winner-prediction baseline
├── train_podium.py             # podium (top-3) model + per-race evaluation
├── tune_hyperparams.py         # time-series cross-validated hyperparameter search
├── walk_forward_eval.py        # multi-year walk-forward comparison of model configs
├── final_comparison.py         # ablation: isolates feature vs. hyperparameter effects
└── README.md
```

## Setup

```bash
pip install pandas scikit-learn xgboost
```

Download the dataset zip from Kaggle, extract it, and place all CSVs inside a `data/` folder at the project root (or adjust the file paths at the top of each script).

## Usage

Run the scripts in this order:

```bash
python build_features.py          # produces model_table.csv
python build_features_v2.py        # produces model_table_v2.csv (extended features)
python train_baseline.py           # winner-prediction baseline (logistic regression + XGBoost)
python train_podium.py             # podium-prediction model with proper per-race evaluation
python tune_hyperparams.py         # time-series CV hyperparameter search
python walk_forward_eval.py        # honest multi-year comparison of configurations
```

Each script prints its results to the console; `build_features*.py` scripts also write out the intermediate CSV feature tables used by everything downstream.

## Methodology

**Era selection.** The model trains on 2000–2024 only. Earlier eras used different points systems, smaller and inconsistent grids, and far less reliable cars, and mixing them in blurs what the model learns rather than helping it.

**Leakage prevention.** Championship standings (`driver_standings.csv`, `constructor_standings.csv`) are cumulative *after* each race, so they're shifted by one round within each season before use, so a race's features only reflect the standings entering that race, never including its own result. Rolling-form features (recent average finishing position) are similarly shifted so a race never has access to its own outcome.

**Chronological splitting.** All train/test splits are by year, never randomly shuffled, since a model that's seen the future shouldn't be evaluated on the past.

**Evaluation.** Because podium finishes are a minority outcome per row (~15% positive rate), raw accuracy is not used. Instead, for each race, the model's top-3 predicted drivers are compared against the actual podium, and the average overlap (0–3) and exact-match rate are reported, alongside AUC. Results are always compared against a naive baseline (top-3 grid starters = predicted podium).

**Features used:** starting grid position, driver age, championship points/position/wins entering the race (driver and constructor, properly shifted), rolling recent form (driver and constructor), qualifying gap to pole in seconds, constructor's combined grid pace for the weekend, rolling DNF/reliability rate, and the driver's historical average finish at that specific circuit.

## Results

On a 2023–2024 holdout (46 races), trained on 2000–2022:

| Model | AUC | Avg. podium overlap (out of 3) | Exact podium match rate |
|---|---|---|---|
| Naive baseline (top-3 grid = podium) | — | 1.67 | 10.9% |
| Podium model (core features) | 0.907 | 1.89 | 19.6% |

A walk-forward evaluation across 7 separate test years (2018–2024) showed that adding extra features (qualifying gap, DNF rate, circuit history) and tuning hyperparameters further produced no statistically meaningful improvement over the simpler core-feature model — all configurations landed within 1.999–2.029 average overlap, well within the year-to-year noise. The simpler model was kept as final for that reason.

## Key lessons from this project

- Leakage hides in cumulative tables like season standings; always check whether a feature reflects the state *before* or *after* the event you're predicting.
- A tree model's missing-value handling only works as well as the missingness pattern at prediction time resembles what it saw in training — it is not a substitute for actually having the data.
- Team and entity renames (e.g. a constructor rebranding) silently break name-based categorical encoding; a stable ID is more robust where one exists.
- A single holdout window can make a perfectly fine change look like a regression (or a bad change look like an improvement) purely from sampling noise. Validate across multiple time windows before trusting any comparison.
- More features and more aggressive tuning are not automatically better, especially on a dataset this size with this much inherent outcome randomness. Prefer the simpler model when performance is statistically indistinguishable.

## Limitations and possible extensions

This project uses only static historical results data — it has no access to weather, tire strategy, practice session pace, or real-time car development trends within a season, all of which materially affect race outcomes in reality. A natural extension would be incorporating session-level telemetry (e.g. via the FastF1 Python package for 2018+ seasons) or weather data per race. The model also cannot make a meaningful prediction before qualifying happens, since starting grid position is by far its strongest feature — a separate no-grid model would be needed for genuine pre-qualifying forecasts, with the understanding that it will be noticeably weaker.

## Credits

Data: Ergast Developer API / Jolpica-F1, as redistributed on Kaggle. This project is for personal/educational practice and is not affiliated with Formula 1, the FIA, or any team.
