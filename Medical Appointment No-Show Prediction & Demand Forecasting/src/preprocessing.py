import pandas as pd
import numpy as np
from pathlib import Path

def load_data(csv_path):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    # try common date columns
    for col in ['appointment_date', 'appointment_date_continuous']:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except Exception:
                pass
    return df


def basic_cleaning(df):
    df = df.copy()
    # normalize column names
    df.columns = [c.strip() for c in df.columns]

    # target to binary if present
    if 'no_show' in df.columns:
        df['no_show'] = df['no_show'].astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0})

    # binary health flags: fill missing with 0
    binary_cols = [c for c in ['Hipertension', 'Diabetes', 'Alcoholism', 'Handcap', 'Scholarship', 'SMSreceived'] if c in df.columns]
    for c in binary_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(int)

    # Numeric columns: age, temps, rain
    if 'age' in df.columns:
        df['age_missing'] = df['age'].isna().astype(int)
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
        df['age'] = df['age'].fillna(df['age'].median())

    # Categorical fills
    for c in df.select_dtypes(include=['object']).columns:
        if c in ['no_show']:
            continue
        df[c] = df[c].astype(str).str.strip().replace({'nan': None})
        df[c] = df[c].fillna('Unknown')

    # Weather numeric fills
    weather_cols = [c for c in df.columns if any(k in c.lower() for k in ['temp', 'rain', 'heat', 'intensity', 'storm'])]
    for c in weather_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        df[c] = df[c].fillna(df[c].median())

    return df


def engineer_features(df):
    df = df.copy()

    # datetime features from appointment_date or appointment_date_continuous
    date_col = None
    if 'appointment_date' in df.columns:
        date_col = 'appointment_date'
    elif 'appointment_date_continuous' in df.columns:
        date_col = 'appointment_date_continuous'

    if date_col is not None:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df['appt_day'] = df[date_col].dt.day
        df['appt_weekday'] = df[date_col].dt.weekday
        df['appt_month'] = df[date_col].dt.month
        df['appt_dayofyear'] = df[date_col].dt.dayofyear
        df['is_weekend'] = df['appt_weekday'].isin([5,6]).astype(int)
        # seasonal cycles
        df['dayofyear_sin'] = np.sin(2 * np.pi * df['appt_dayofyear'] / 365)
        df['dayofyear_cos'] = np.cos(2 * np.pi * df['appt_dayofyear'] / 365)

    # age groups
    if 'age' in df.columns:
        df['under_12'] = (df['age'] < 12).astype(int)
        df['over_60'] = (df['age'] >= 60).astype(int)

    # appointment shift: if present keep, else derive from time
    if 'appointment_shift' not in df.columns and 'appointment_time' in df.columns:
        # try parse hour
        try:
            times = pd.to_datetime(df['appointment_time'], errors='coerce')
            hours = times.dt.hour
            df['appointment_shift'] = pd.cut(hours, bins=[-1,11,17,24], labels=['morning','afternoon','evening'])
        except Exception:
            df['appointment_shift'] = 'Unknown'

    # interaction features: temperature * rain if present
    temp_cols = [c for c in df.columns if 'temp' in c.lower()]
    rain_cols = [c for c in df.columns if 'rain' in c.lower()]
    if temp_cols and rain_cols:
        df['temp_x_rain'] = df[temp_cols[0]] * df[rain_cols[0]]

    # encode simple categorical counts: specialty/place frequency
    for c in ['specialty', 'place']:
        if c in df.columns:
            vc = df[c].value_counts(dropna=False)
            df[f'{c}_freq'] = df[c].map(vc).fillna(0).astype(int)

    return df


def save_processed(df, dest_path):
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    return dest


def run_pipeline(raw_csv, out_csv):
    df = load_data(raw_csv)
    df = basic_cleaning(df)
    df = engineer_features(df)
    saved = save_processed(df, out_csv)
    return df, saved


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', default='../data/medical_appointments.csv')
    parser.add_argument('--out', default='../data/processed_medical_appointments.csv')
    args = parser.parse_args()
    df, saved = run_pipeline(args.raw, args.out)
    print('Processed saved to', saved)
