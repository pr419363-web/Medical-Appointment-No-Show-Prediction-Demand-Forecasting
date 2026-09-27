# Medical Appointment No-Show Prediction & Demand Forecasting

## Project Overview

This project develops two complementary machine learning systems for the University of Vale do Itajaí Center of Specialization in Physical and Intellectual Rehabilitation (CER):

1. **No-Show Prediction (Binary Classification)**: Identifies patients at risk of missing appointments
2. **Demand Forecasting (Time Series Regression)**: Predicts daily appointment volumes

## Problem Statement

- **Historical No-Show Rate**: 31.78% in the supplied CSV
- **Impact**: Revenue loss, wasted specialist capacity, inefficient staffing
- **Goal**: Reduce no-shows through early identification and optimize staff scheduling through demand prediction

## Dataset

- **Size**: 109,593 rows × 26 columns
- **Target**: `no_show` (yes/no)
- **Features**:
  - Patient demographics (gender, age, disability, needs_companion)
  - Appointment details (specialty, time, shift, date)
  - Location (`place`); 26,289 distinct non-null strings in the supplied CSV, so a 13-city mapping is not established
  - Health conditions (Hypertension, Diabetes, Alcoholism, etc.)
  - Weather (temperature, rainfall, intensity, storm indicators)
  - SMS received flag

**Data Location**: `data/raw/medical_appointments.csv` (also available from [Google Drive](https://drive.google.com/file/d/1NdUpKIZwv1uBIszx_8x53NaS1FE9nT_O/view?usp=sharing))

The source CSV has 26 columns. See the [data dictionary](data/data_dictionary.md) for the source names, types, and definitions.

## Project Structure

```
.
├── data/                           # Data files
│   ├── raw/                        # Original dataset
│   └── processed/                  # Preprocessed data
├── notebooks/                      # Jupyter notebooks
│   ├── 01_EDA.ipynb               # Exploratory data analysis
│   ├── 02_Preprocessing.ipynb     # Data cleaning & feature engineering
│   ├── 03_NoShow_Prediction.ipynb # Classification model development
│   └── 04_Demand_Forecasting.ipynb # Time series model development
├── src/                            # Source code modules
│   ├── preprocessing.py           # Data preprocessing utilities
│   ├── feature_engineering.py     # Feature creation
│   ├── evaluation.py              # Evaluation metrics
│   ├── run_preprocessing.py       # Build processed train/test splits
│   ├── train_forecasting.py       # Forecast model training experiment
│   └── evaluate_baselines.py      # Reproducible held-out baseline reports
├── app/                            # Streamlit application
│   ├── streamlit_app.py          # Main app
│   └── config.py                  # App configuration
├── models/                        # Preprocessing artifacts; no inference models saved
├── reports/                       # Generated metrics and evaluation figures
├── data/data_dictionary.md        # Raw CSV schema
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Installation & Setup

### 1. Clone/Download the Repository
```bash
cd "Medical Appointment No-Show Prediction & Demand Forecasting"
```

### 2. Create Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Dataset
- Download from the [Google Drive link](https://drive.google.com/file/d/1NdUpKIZwv1uBIszx_8x53NaS1FE9nT_O/view?usp=sharing)
- Place the CSV file in `data/raw/` folder

## Usage

### Run Exploratory Data Analysis
```bash
jupyter notebook notebooks/01_EDA.ipynb
```

### Reproduce Baseline Evaluation
The script fits a fixed-seed classifier on the saved train/test split and a time-ordered daily-demand baseline on the raw CSV. It writes metrics, held-out predictions, and plots to `reports/`.

```bash
python src/evaluate_baselines.py
```

This evaluation is separate from the Streamlit form: it does not create a deployed inference model.

### Launch Streamlit Application
```bash
streamlit run app/streamlit_app.py
```

The application will open at `http://localhost:8501`. Its input screens are interface previews; prediction and future-forecast serving are disabled until trained inference models are added.

## Evaluation Results

The following are measured baselines, not target values. Reproduce them with `python src/evaluate_baselines.py`.

| Task | Baseline | Held-out results | Goal | Status |
|------|----------|------------------|------|--------|
| No-show classification | Random Forest, fixed seed 42, existing stratified split | F1 = 0.5617; ROC-AUC = 0.7671 | F1 > 0.70; ROC-AUC > 0.75 | F1 goal not met; ROC-AUC goal met |
| Daily demand | HistGradientBoostingRegressor, fixed seed 42, chronological 80/20 split | MAPE = 5,008.82%; R² = -0.0033; MAE = 210.55 appointments/day | MAPE < 20%; R² > 0.65 | Goals not met |

The classifier confusion matrix (actual rows, predicted columns; order: show, no-show) is `[[11725, 3227], [2986, 3981]]`. The demand MAPE is especially sensitive to low-volume days; the held-out period contains 94 days. Treat both results as baselines requiring improvement, not evidence of production readiness.

### Evaluation Figures

The reproducible evaluation script generates the confusion matrix, classifier feature-importance chart, and held-out demand forecast plot under `reports/figures/`.

![No-show baseline confusion matrix](reports/figures/confusion_matrix.png)

![No-show baseline feature importance](reports/figures/feature_importance.png)

![Held-out daily demand forecast](reports/figures/demand_forecast_backtest.png)

### Streamlit Screenshots

The Streamlit screens are interface previews; scores and future demand estimates are not currently served by the app.

![No-show risk prediction form](reports/screenshots/no_show_form.png)

![Demand forecasting dashboard](reports/screenshots/demand_forecasting_dashboard.png)

## Key Features

### No-Show Predictor
- Input form preview for patient and appointment details
- No individual risk score is served; inference model packaging remains to be done
- Evaluation feature-importance chart is generated under `reports/figures/`

### Demand Forecaster
- Time period, specialty, and location-label controls shown as a UI preview
- No future forecast or confidence interval is served
- Evaluation currently backtests overall daily appointment counts only

## Business Use Cases

1. **Risk-Based Patient Engagement**: SMS reminders for high-risk patients
2. **Intelligent Staffing Optimization**: Forecast daily volume for shift planning
3. **Revenue Protection**: Reduce no-show losses through early intervention
4. **Geographic Resource Planning**: Normalize and validate the high-cardinality `place` field before use
5. **Seasonal & Weather Adaptations**: Account for environmental factors
6. **Health-Based Segmentation**: Tailor strategies by health conditions
7. **Specialty-Level Demand Planning**: Future extension; not evaluated by the current overall-demand baseline

## Business Impact (Hypotheses)

The dataset's historical no-show rate is about 31.8%. As an illustrative, unvalidated scenario, a 10% relative reduction from that baseline would mean roughly 210 fewer no-shows in an average 30.4-day month at the observed appointment volume. This is not a measured intervention effect; validate reminder strategies with a controlled pilot.

Scheduling-efficiency improvement has not been measured. Track a pre/post or controlled-pilot KPI before making a percentage claim.

## Data Dictionary

The [data dictionary](data/data_dictionary.md) documents every field in the raw CSV, including its exact source spelling and nullable values. No SQL schema or database integration is currently implemented.

## Generated Evaluation Outputs

Running the baseline script creates `reports/metrics.json`, held-out forecast predictions, and the three figures shown above. It does not save deployable models.

## Troubleshooting

### Missing or incompatible evaluation files
The classifier baseline expects the processed train/test files in `data/processed/`. Regenerate them with `python src/run_preprocessing.py` if they are missing.

### Class Imbalance
The dataset has ~31.8% no-shows. Strategies:
- SMOTE oversampling
- Class weight adjustments
- Threshold tuning
- F1-score optimization

## Contributing

Document all preprocessing steps and model changes in notebooks.
Save trained models using joblib for reproducibility.

## License

Educational project for healthcare analytics.

## Contact

For questions about the project structure or setup, refer to the individual notebook documentation.

---

**Last Updated**: September 2026
**Status**: Baseline evaluation documented; targets not yet met
