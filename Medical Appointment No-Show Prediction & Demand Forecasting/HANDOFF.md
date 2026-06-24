# Handoff Instructions

This document summarizes how to reproduce, run, and hand over the Medical Appointment No-Show Prediction & Demand Forecasting project.

Prerequisites

- Python 3.9+ (3.10 recommended)
- Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Quick reproduction steps

1. Download raw dataset from the provided Google Drive link and place it at `data/medical_appointments.csv`. Alternatively, run `notebooks/01_EDA.ipynb` which attempts to download automatically.

2. Run preprocessing to create processed data:

```powershell
python src/preprocessing.py --raw data/medical_appointments.csv --out data/processed_medical_appointments.csv
```

Or open `notebooks/02_Preprocessing_Feature_Engineering.ipynb` and run the cells.

3. Train no-show models (notebooks present):

```powershell
# Run inside the venv
jupyter notebook notebooks/03_NoShow_Modeling.ipynb
```

4. Train demand forecasting models:

```powershell
jupyter notebook notebooks/04_Demand_Forecasting.ipynb
```

5. Evaluate and produce final metrics and sample predictions:

```powershell
jupyter notebook notebooks/05_Evaluation_and_Handoff.ipynb
```

6. Run Streamlit app locally:

```powershell
streamlit run app.py
```

Release bundle

- Use `python scripts/collect_models.py` to copy models, notebooks, and docs into `release/` for distribution.

Operational notes

- The Streamlit app expects processed data at `data/processed_medical_appointments.csv` and models in `models/`.
- Prediction latency should be under 2 seconds for the no-show model when running on a standard laptop.
- Retrain cadence: recommend weekly or monthly depending on data drift.

Contact

For questions, refer to `docs/MODEL_CARD.md` and `docs/DATA_DICTIONARY.md`, or open the notebooks for reproducibility details.
