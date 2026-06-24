# Model Card — Medical Appointment No-Show & Demand Forecasting

## Overview
This repository contains two model families:

- No-show classification models (binary): `no_show_best.joblib` (best-performing pipeline). Also saved: `no_show_logistic.joblib`, `no_show_random_forest.joblib`, `no_show_xgboost.joblib` (if trained).
- Demand forecasting models (daily counts): `demand_lgbm.pkl`, `demand_sarimax.pkl`, `demand_lstm.keras` (if trained).

Models are saved under the `models/` directory after training.

## Purpose
- No-show model: predict whether a scheduled patient will miss an appointment (1 = no-show). Outputs a probability score used to trigger interventions (SMS reminders, call-outs).
- Demand forecast: predict daily appointment volume (global or by specialty when aggregated accordingly).

## Intended Use
- Operational decision support for clinic staff and schedulers.
- Not for individual clinical decision-making.

## Performance Targets
- No-show classifier: F1 > 0.70, ROC-AUC > 0.75 on held-out test set
- Demand forecasting: MAPE < 20%, R² > 0.65 on test set

## Files and How to Load
- Example (no-show):

```python
import joblib
pipe = joblib.load('models/no_show_best.joblib')
proba = pipe.predict_proba(X_new)[:,1]
```

- Example (LightGBM demand):

```python
import joblib
model = joblib.load('models/demand_lgbm.pkl')
preds = model.predict(X_lag_features)
```

- SARIMAX: loaded via pickle (statsmodels results object).
- LSTM: saved in Keras `.keras` format; load with `tf.keras.models.load_model`.

## Caveats and Limitations
- Models are trained on historical appointment patterns from CER and may not generalize to other clinics without retraining.
- Weather and environmental features must be aligned to the appointment date for correct forecasts.
- Class imbalance in no-show target requires careful thresholding and calibration before operational use.

## Retraining
- Use the notebooks: `notebooks/03_NoShow_Modeling.ipynb` and `notebooks/04_Demand_Forecasting.ipynb` to reproduce training and create the `models/` artifacts.
- Save final artifacts to `models/` and use `scripts/collect_models.py` to generate a `release/` bundle.

## Contact
For questions about model training hyperparameters or dataset provenance, see `README.md` and the notebooks in `notebooks/`.
