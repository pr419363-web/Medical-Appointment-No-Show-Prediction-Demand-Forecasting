import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import pickle

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

try:
    import lightgbm as lgb
except Exception:
    lgb = None

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
except Exception:
    tf = None

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except Exception:
    SARIMAX = None


def load_processed(path='data/processed_medical_appointments.csv'):
    p = Path(path)
    if not p.exists():
        p = Path('..') / path
        if not p.exists():
            raise FileNotFoundError(f"Processed CSV not found at {path}")
    df = pd.read_csv(p, parse_dates=['appointment_date'], low_memory=False)
    return df


def aggregate_daily(df, date_col='appointment_date', group_cols=None):
    df = df.copy()
    if date_col not in df.columns:
        # try appointment_date_continuous
        if 'appointment_date_continuous' in df.columns:
            df[date_col] = pd.to_datetime(df['appointment_date_continuous'], errors='coerce')
        else:
            raise ValueError('No date column found for aggregation')

    df['date'] = pd.to_datetime(df[date_col]).dt.date
    if group_cols:
        grouped = df.groupby(['date'] + group_cols).size().reset_index(name='count')
    else:
        grouped = df.groupby(['date']).size().reset_index(name='count')
    grouped['date'] = pd.to_datetime(grouped['date'])
    grouped = grouped.sort_values('date').set_index('date')
    return grouped


def create_lag_features(series, lags=[1,7,14], rolling_windows=[7,14]):
    df = pd.DataFrame({'y': series})
    for l in lags:
        df[f'lag_{l}'] = df['y'].shift(l)
    for w in rolling_windows:
        df[f'roll_mean_{w}'] = df['y'].shift(1).rolling(w).mean()
        df[f'roll_std_{w}'] = df['y'].shift(1).rolling(w).std()
    df = df.dropna()
    return df


def train_lightgbm(df_ts, test_size_days=30, params=None, save_dir='models'):
    if lgb is None:
        raise ImportError('lightgbm is not installed')
    series = df_ts['count']
    features = create_lag_features(series)
    X = features.drop(columns=['y'])
    y = features['y']

    # chronological split
    split_idx = len(X) - test_size_days
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    lgb_train = lgb.Dataset(X_train, label=y_train)
    params = params or {'objective': 'regression', 'metric': 'rmse', 'verbosity': -1}
    gbm = lgb.train(params, lgb_train, num_boost_round=100)

    preds = gbm.predict(X_test)
    metrics = evaluate_forecast(y_test, preds)

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(gbm, Path(save_dir) / 'demand_lgbm.pkl')
    return metrics


def train_sarimax(df_ts, order=(1,1,1), seasonal_order=(0,0,0,0), test_size_days=30, save_dir='models'):
    if SARIMAX is None:
        raise ImportError('statsmodels SARIMAX not available')
    series = df_ts['count']
    train = series.iloc[:-test_size_days]
    test = series.iloc[-test_size_days:]

    model = SARIMAX(train, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    preds = res.forecast(steps=len(test))
    metrics = evaluate_forecast(test, preds)

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(save_dir) / 'demand_sarimax.pkl', 'wb') as f:
        pickle.dump(res, f)
    return metrics


def train_lstm(df_ts, test_size_days=30, epochs=20, batch_size=16, save_dir='models'):
    if tf is None:
        raise ImportError('tensorflow is not available')
    series = df_ts['count']
    features = create_lag_features(series, lags=[1,2,3,7,14], rolling_windows=[7])
    X = features.drop(columns=['y']).values
    y = features['y'].values

    split_idx = len(X) - test_size_days
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # reshape for LSTM: (samples, timesteps, features) - treat each row as timestep=1
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    model = Sequential()
    model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=1)

    preds = model.predict(X_test).flatten()
    metrics = evaluate_forecast(y_test, preds)

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    model.save(Path(save_dir) / 'demand_lstm.keras')
    return metrics


def evaluate_forecast(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-8, None))) * 100
    r2 = r2_score(y_true, y_pred)
    return {'mae': mae, 'rmse': rmse, 'mape': mape, 'r2': r2}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed', default='../data/processed_medical_appointments.csv')
    parser.add_argument('--out', default='../models')
    args = parser.parse_args()

    df = load_processed(args.processed)
    daily = aggregate_daily(df)
    print('Data prepared with', len(daily), 'days')

    print('Training LightGBM...')
    try:
        print(train_lightgbm(daily, save_dir=args.out))
    except Exception as e:
        print('LightGBM training skipped:', e)

    print('Training SARIMAX...')
    try:
        print(train_sarimax(daily, save_dir=args.out))
    except Exception as e:
        print('SARIMAX training skipped:', e)

    print('Training LSTM...')
    try:
        print(train_lstm(daily, save_dir=args.out))
    except Exception as e:
        print('LSTM training skipped:', e)
