# Data Dictionary

This file summarizes the main dataset columns used in the project. The original dataset (109,593 rows × 26 columns) contains appointment-level records.

- `no_show`: Target — whether the patient missed the appointment (`yes`/`no` mapped to 1/0).
- `gender`: Patient gender (F/M).
- `age`: Patient age in years. Missing values handled via median imputation. Derived flags: `under_12`, `over_60`.
- `disability` / `Handcap`: Disability indicator (binary or categorical).
- `needs_companion`: Whether patient requires a companion for appointments.
- `specialty`: Rehabilitation specialty (e.g., physiotherapy, psychology, speech therapy).
- `appointment_time`: Time of appointment (string/H:M).
- `appointment_shift`: Categorical (morning/afternoon/evening). Derived when missing.
- `appointment_date_continuous` / `appointment_date`: Date of appointment (continuous calendar).
- `place`: City/location where appointment occurs (13 cities in service region).
- `Hipertension`: Binary flag for hypertension diagnosis.
- `Diabetes`: Binary flag for diabetes diagnosis.
- `Alcoholism`: Binary flag for alcoholism.
- `Scholarship`: Indicator for scholarship/benefit.
- `SMSreceived`: Whether patient received an SMS reminder (binary).

Weather / Environmental features (may be missing for some rows):
- `avg_temp`, `max_temp` (or similar): Temperature measurements.
- `rain` or `avg_rain`: Precipitation measures.
- `heat_intensity`, `rain_intensity`: Derived intensity indicators.
- `rainy_day_before`, `storm_day_before`: Binary flags for recent adverse weather.

Notes on missingness and preprocessing:
- `age` ~21% missing — imputed with median and tracked with `age_missing` flag.
- `specialty` ~18% missing — filled with `Unknown` or treated as separate category.
- `disability` / `Handcap` ~15% missing — treated as `Unknown` or imputed as 0 depending on context.
- `place` ~10.5% missing — filled with `Unknown`.
- Weather ~2% missing — imputed with median values.

Feature engineering summary:
- Temporal features: `appt_weekday`, `appt_month`, `is_weekend`, `dayofyear_sin`, `dayofyear_cos`.
- Interaction: `temp_x_rain` when both temp and rain available.
- Frequency features: `specialty_freq`, `place_freq`.
- Lag/rolling features (forecasting): `lag_1`, `lag_7`, `lag_14`, `roll_mean_7`, etc.

See `notebooks/01_EDA.ipynb` and `notebooks/02_Preprocessing_Feature_Engineering.ipynb` for exact preprocessing steps and code.
