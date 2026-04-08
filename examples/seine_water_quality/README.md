# Seine Estuary — Water Quality Reconstruction

This example reproduces the experimental setup of:

> **Janbain, I., Jardani, A., Deloffre, J., Massei, N. (2023).**
> *Deep Learning Approaches for Numerical Modeling and Historical
> Reconstruction of Water Quality Parameters in Lower Seine.*
> Water (MDPI), 15(9), 1773. https://doi.org/10.3390/w15091773

## Task

Use **water level** measurements at five stations along the Lower Seine
(Honfleur → Rouen) as input to a deep recurrent model that predicts and
reconstructs three water-quality parameters at Tancarville:

- electrical conductivity (μS·cm⁻¹)
- dissolved oxygen (mg·L⁻¹)
- turbidity (NTU)

Water-level sensors are cheap and continuous. Water-quality sensors are
expensive and prone to long gaps. Learning the mapping between the two
lets us reconstruct decades of historical water-quality data from
water-level archives alone.

## Data

The original dataset is hourly, 1990–2022, and is **not redistributed
here** because it is owned by the M2C laboratory and partner agencies
(GIP Seine-Aval, DREAL, EDF). To request access, contact the M2C
laboratory directly.

The repository ships a **synthetic stand-in** with the same shape and
similar tidal correlation structure so the full pipeline can be exercised
on a fresh clone — see `scripts/generate_synthetic_demo.py` and
`configs/synthetic_demo.yaml`.

## Reproducing the headline result

```bash
# 1. Place the real Seine CSV at data/seine_hourly.csv
# 2. Hyper-parameter search + final retraining
python scripts/tune.py --config configs/seine_water_quality.yaml

# 3. Use the trained model to fill gaps in any new file
python scripts/reconstruct.py \
    --config configs/seine_water_quality.yaml \
    --model outputs/seine_water_quality/model.keras \
    --input data/seine_with_gaps.csv \
    --output outputs/seine_water_quality/reconstructed.csv
```

Outputs (under `outputs/seine_water_quality/`):

- `model.keras` — best Keras model
- `x_scaler.joblib`, `y_scaler.joblib` — fitted scalers (needed for inference)
- `best_params.json` — Optuna's best trial
- `optuna_trials.csv` — full search history
- `metrics.json` — RMSE / MAE / R² / NSE / KGE on the held-out test set
- `training_history.png`, `predictions.png`, `scatter.png`
