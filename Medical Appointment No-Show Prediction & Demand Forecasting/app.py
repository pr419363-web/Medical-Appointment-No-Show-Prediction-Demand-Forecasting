import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import datetime

BASE = Path(__file__).parent
DATA_PATH = BASE / 'data' / 'processed_medical_appointments.csv'
MODELS_DIR = BASE / 'models'

st.set_page_config(page_title="No-Show & Demand Forecast", layout="wide")
st.title("Medical Appointment No-Show Prediction & Demand Forecasting")

@st.cache_data
def load_processed_df(path):
    p = Path(path)
    if not p.exists():
        return None
    df = pd.read_csv(p, low_memory=False, parse_dates=[col for col in ['appointment_date', 'appointment_date_continuous'] if col in pd.read_csv(p,nrows=0).columns])
    return df

@st.cache_resource
def load_models():
    models = {}
    no_show_path = MODELS_DIR / 'no_show_best.joblib'
    if no_show_path.exists():
        try:
            models['no_show'] = joblib.load(no_show_path)
        except Exception:
            models['no_show'] = None
    else:
        models['no_show'] = None

    # demand models
    lgb_path = MODELS_DIR / 'demand_lgbm.pkl'
    sarimax_path = MODELS_DIR / 'demand_sarimax.pkl'
    lstm_path = MODELS_DIR / 'demand_lstm.keras'
    models['demand_lgbm'] = joblib.load(lgb_path) if lgb_path.exists() else None
    models['demand_sarimax'] = None
    if sarimax_path.exists():
        try:
            import pickle
            with open(sarimax_path, 'rb') as f:
                models['demand_sarimax'] = pickle.load(f)
        except Exception:
            models['demand_sarimax'] = None
    models['demand_lstm'] = lstm_path if lstm_path.exists() else None
    return models

processed_df = load_processed_df(DATA_PATH)
models = load_models()

tab1, tab2 = st.tabs(["No-Show Predictor", "Demand Forecaster"])

with tab1:
    st.header("No-Show Predictor")
    st.write("Enter patient & appointment details and get a no-show risk score.")

    if processed_df is None:
        st.warning('Processed data not found at data/processed_medical_appointments.csv. Run preprocessing first.')
    else:
        # derive feature template from processed data
        X_cols = [c for c in processed_df.columns if c not in ['no_show', 'appointment_date', 'appointment_time', 'appointment_date_continuous']]
        numeric_cols = processed_df.select_dtypes(include=['number']).columns.tolist()
        medians = processed_df[numeric_cols].median()

        with st.form('no_show_form'):
            col1, col2 = st.columns(2)
            with col1:
                gender = st.selectbox('Gender', options=['Unknown','F','M'])
                age = st.number_input('Age', min_value=0, max_value=120, value=processed_df['age'].median() if 'age' in processed_df.columns else 30)
                specialty = st.text_input('Specialty', value='Unknown')
                place = st.text_input('Place (city)', value='Unknown')
                appointment_date = st.date_input('Appointment date', value=datetime.date.today())
            with col2:
                appointment_time = st.text_input('Appointment time (HH:MM)', value='09:00')
                appointment_shift = st.selectbox('Appointment shift', options=['Unknown','morning','afternoon','evening'])
                hipertension = st.checkbox('Hipertension')
                diabetes = st.checkbox('Diabetes')
                alcoholism = st.checkbox('Alcoholism')
                handicap = st.checkbox('Handcap')

            sms = st.checkbox('SMS received')
            submitted = st.form_submit_button('Predict risk')

        if submitted:
            # build an input dataframe with expected columns
            row = {c: None for c in X_cols}
            # numeric fills
            for c in numeric_cols:
                if c == 'age':
                    row['age'] = age
                else:
                    row[c] = float(medians.get(c, 0))

            # categorical fills
            if 'gender' in X_cols:
                row['gender'] = gender
            if 'specialty' in X_cols:
                row['specialty'] = specialty
            if 'place' in X_cols:
                row['place'] = place
            if 'appointment_shift' in X_cols:
                row['appointment_shift'] = appointment_shift

            for col_name, val in [('Hipertension', hipertension), ('Diabetes', diabetes), ('Alcoholism', alcoholism), ('Handcap', handicap), ('SMSreceived', sms)]:
                if col_name in X_cols:
                    row[col_name] = int(bool(val))

            input_df = pd.DataFrame([row])

            if models.get('no_show') is None:
                st.error('No trained no-show model found at models/no_show_best.joblib')
            else:
                with st.spinner('Predicting...'):
                    try:
                        proba = models['no_show'].predict_proba(input_df)[:,1][0]
                        risk_pct = float(proba) * 100
                        st.metric('No-show risk', f"{risk_pct:.1f}%")
                        st.write('Probability score:', proba)
                    except Exception as e:
                        st.error(f'Prediction failed: {e}')

with tab2:
    st.header('Demand Forecaster')
    st.write('View historical daily appointment counts and a short-term forecast.')

    if processed_df is None:
        st.warning('Processed data not found. Run preprocessing first.')
    else:
        # aggregate daily counts
        df = processed_df.copy()
        date_col = 'appointment_date' if 'appointment_date' in df.columns else 'appointment_date_continuous'
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        daily = df.groupby(df[date_col].dt.date).size().reset_index(name='count')
        daily['date'] = pd.to_datetime(daily[date_col]) if date_col in daily.columns else pd.to_datetime(daily['index'])
        daily = daily[['date','count']].set_index('date').sort_index()
        st.line_chart(daily['count'])

        horizon = st.number_input('Forecast horizon (days)', min_value=1, max_value=60, value=14)
        if models.get('demand_lgbm') is not None:
            with st.spinner('Generating forecast with LightGBM...'):
                try:
                    gbm = models['demand_lgbm']
                    series = daily['count'].copy()
                    # create lag features for last available day
                    def make_features_for_forecast(series, lags=[1,7,14], rolling_windows=[7,14]):
                        s = series.copy()
                        features = []
                        for h in range(horizon):
                            vals = {}
                            for l in lags:
                                idx = len(s) - l
                                vals[f'lag_{l}'] = s.iloc[idx] if idx>=0 else 0
                            for w in rolling_windows:
                                vals[f'roll_mean_{w}'] = s.shift(1).rolling(w).mean().iloc[-1]
                                vals[f'roll_std_{w}'] = s.shift(1).rolling(w).std().iloc[-1]
                            features.append(vals)
                            # append predicted placeholder (will be filled after predicting)
                        return pd.DataFrame(features)

                    feat_df = make_features_for_forecast(series)
                    preds = gbm.predict(feat_df)
                    last_date = daily.index.max()
                    dates = [last_date + pd.Timedelta(days=i+1) for i in range(horizon)]
                    forecast = pd.Series(preds, index=dates)
                    combined = pd.concat([daily['count'], forecast.rename('forecast')], axis=0)
                    st.line_chart(pd.DataFrame({'history': daily['count'], 'forecast': forecast}))
                    st.table(pd.DataFrame({'date': dates, 'forecast_count': preds.round(1)}))
                except Exception as e:
                    st.error('Forecasting failed: ' + str(e))
        else:
            st.info('No LightGBM demand model found. Showing simple rolling-mean baseline.')
            last = daily['count'].iloc[-14:]
            baseline = [last.mean()] * horizon
            last_date = daily.index.max()
            dates = [last_date + pd.Timedelta(days=i+1) for i in range(horizon)]
            st.line_chart(pd.DataFrame({'history': daily['count'], 'baseline_forecast': pd.Series(baseline, index=dates)}))

st.sidebar.markdown("### Project: Medical Appointment No-Show Prediction & Demand Forecasting")
st.sidebar.markdown("Data-driven tools for CER operations")
