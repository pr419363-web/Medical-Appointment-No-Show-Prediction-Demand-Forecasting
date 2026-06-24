# Medical Appointment No-Show Prediction & Demand Forecasting

Overview

This repository contains code and notebooks for predicting patient no-shows and forecasting daily appointment demand for the University of Vale do Itajaí Center of Specialization in Physical and Intellectual Rehabilitation (CER).

Structure

- `data/` - raw dataset (not checked in)
- `notebooks/` - analysis and modeling notebooks
- `src/` - helper modules and model training scripts
- `models/` - saved trained models (.pkl/.joblib)
- `app.py` - Streamlit app (no-show predictor + demand forecaster)
- `requirements.txt` - Python dependencies

Models and Documentation

- After training, models are saved to `models/`. The recommended artifact names are:
	- `models/no_show_best.joblib` — best no-show classifier pipeline
	- `models/demand_lgbm.pkl` — LightGBM demand model
	- `models/demand_sarimax.pkl` — SARIMAX model (pickled results)
	- `models/demand_lstm.keras` — LSTM Keras model (optional)

- Use `scripts/collect_models.py` to package trained models, notebooks and docs into `release/` for sharing.

Documentation

- See `docs/MODEL_CARD.md` and `docs/DATA_DICTIONARY.md` for model descriptions and the dataset schema.

Quick start

1. Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

2. Download the dataset and place it at `data/medical_appointments.csv` (or update paths in notebooks).

3. Run the Streamlit app locally:

```powershell
streamlit run app.py
```

Notebooks

Start with `notebooks/01_EDA.ipynb` for initial exploration and preprocessing steps. See notebooks for modeling and evaluation details.

Notes

- Follow chronological train-test splits for time-series forecasting.
- Save final models to `models/` and load them in `app.py` for inference.

Contact

If you want me to continue, I can implement preprocessing notebooks, model training scripts, and the Streamlit UI.
