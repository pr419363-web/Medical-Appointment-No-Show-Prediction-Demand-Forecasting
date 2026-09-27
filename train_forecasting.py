"""
Complete Time Series Demand Forecasting Model Training
Trains multiple time series models and saves the best one
"""

import pandas as pd
import numpy as np
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler

project_root = r"c:\Users\HP\OneDrive\Desktop\Medical Appointment No-Show Prediction & Demand Forecasting"
raw_data_dir = os.path.join(project_root, 'data', 'raw')
models_dir = os.path.join(project_root, 'models')

def train_time_series_models():
    """Train time series forecasting models"""
    
    print("="*80)
    print("TIME SERIES FORECASTING MODEL TRAINING")
    print("="*80)
    
    # Load raw data
    print("\nLoading data...")
    df = pd.read_csv(os.path.join(raw_data_dir, 'medical_appointments.csv'))
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
    df['appointment_date_continuous'] = pd.to_datetime(df['appointment_date_continuous'])
    
    # Prepare time series data by grouping daily appointments
    print("\nPreparing time series data...")
    daily_data = df.groupby(df['appointment_date_continuous'].dt.date).size().reset_index()
    daily_data.columns = ['date', 'total_appointments']
    daily_data['date'] = pd.to_datetime(daily_data['date'])
    daily_data = daily_data.sort_values('date').reset_index(drop=True)
    
    print(f"Time series data: {daily_data.shape[0]} days")
    print(f"Average daily appointments: {daily_data['total_appointments'].mean():.0f}")
    
    # Feature engineering for time series
    print("\nEngineering time series features...")
    daily_data['day_of_week'] = daily_data['date'].dt.dayofweek
    daily_data['month'] = daily_data['date'].dt.month
    daily_data['quarter'] = daily_data['date'].dt.quarter
    daily_data['week_of_year'] = daily_data['date'].dt.isocalendar().week
    daily_data['is_weekend'] = (daily_data['day_of_week'] >= 5).astype(int)
    
    # Create lag features
    for lag in [1, 7, 30]:
        daily_data[f'lag_{lag}'] = daily_data['total_appointments'].shift(lag)
    
    # Create rolling averages
    daily_data['rolling_mean_7'] = daily_data['total_appointments'].rolling(7).mean()
    daily_data['rolling_mean_30'] = daily_data['total_appointments'].rolling(30).mean()
    
    # Drop NaN values
    daily_data = daily_data.dropna().reset_index(drop=True)
    
    print(f"Feature-engineered data: {daily_data.shape[0]} days, {daily_data.shape[1]} features")
    
    # Train-test split (chronological 80/20)
    print("\nTrain-test split (chronological 80/20)...")
    train_size = int(0.8 * len(daily_data))
    train_data = daily_data[:train_size]
    test_data = daily_data[train_size:]
    
    print(f"Training: {len(train_data)} days")
    print(f"Testing: {len(test_data)} days")
    
    # Select features
    feature_cols = ['day_of_week', 'month', 'quarter', 'week_of_year', 'is_weekend',
                    'lag_1', 'lag_7', 'lag_30', 'rolling_mean_7', 'rolling_mean_30']
    
    X_train = train_data[feature_cols].values
    y_train = train_data['total_appointments'].values
    X_test = test_data[feature_cols].values
    y_test = test_data['total_appointments'].values
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Handle any NaN values from scaling
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    X_train_scaled = imputer.fit_transform(X_train_scaled)
    X_test_scaled = imputer.transform(X_test_scaled)
    
    # Model definitions
    models = {
        'XGBoost': xgb.XGBRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=42, verbosity=0
        ),
        'LightGBM': lgb.LGBMRegressor(
            n_estimators=100, max_depth=7, learning_rate=0.1,
            random_state=42, verbosity=-1
        )
    }
    
    results = {}
    trained_models = {}
    
    # Train and evaluate models
    print("\n" + "-"*80)
    print("TRAINING MODELS")
    print("-"*80)
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train
        model.fit(X_train_scaled, y_train)
        trained_models[name] = model
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        
        # Metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R2': r2,
            'Model': model
        }
        
        print(f"  RMSE:   {rmse:.2f}")
        print(f"  MAE:    {mae:.2f}")
        print(f"  MAPE:   {mape:.2f}%")
        print(f"  R2:     {r2:.4f}")
    
    # Compare models
    print("\n" + "-"*80)
    print("MODEL COMPARISON")
    print("-"*80)
    
    results_df = pd.DataFrame({
        name: {
            'RMSE': results[name]['RMSE'],
            'MAE': results[name]['MAE'],
            'MAPE': results[name]['MAPE'],
            'R2': results[name]['R2']
        }
        for name in results
    }).T
    
    print("\n", results_df.round(4))
    
    # Select best model
    print("\n" + "-"*80)
    print("MODEL SELECTION")
    print("-"*80)
    
    best_name = min(results, key=lambda x: results[x]['MAPE'])
    best_model = results[best_name]['Model']
    best_mape = results[best_name]['MAPE']
    best_r2 = results[best_name]['R2']
    
    print(f"\nBest Model: {best_name}")
    print(f"  MAPE: {best_mape:.2f}% (target: < 20%)")
    print(f"  R2:   {best_r2:.4f} (target: > 0.65)")
    
    if best_mape > 20:
        print(f"  WARNING: MAPE above target!")
    if best_r2 < 0.65:
        print(f"  WARNING: R2 below target!")
    
    # Feature importance
    if hasattr(best_model, 'feature_importances_'):
        print(f"\nTop 10 Important Features:")
        importances = best_model.feature_importances_
        top_indices = np.argsort(importances)[-10:][::-1]
        for i, idx in enumerate(top_indices, 1):
            print(f"  {i:2d}. {feature_cols[idx]:20s} {importances[idx]:.4f}")
    
    # Save models
    print("\n" + "-"*80)
    print("SAVING MODELS")
    print("-"*80)
    
    # Save best model
    best_model_path = os.path.join(models_dir, 'demand_forecaster.joblib')
    joblib.dump(best_model, best_model_path)
    print(f"\nBest model saved: {best_model_path}")
    
    # Save scaler
    scaler_path = os.path.join(models_dir, 'ts_scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved: {scaler_path}")
    
    # Save all models
    all_models_path = os.path.join(models_dir, 'all_forecasting_models.joblib')
    joblib.dump(trained_models, all_models_path)
    print(f"All models saved: {all_models_path}")
    
    # Save results
    results_path = os.path.join(models_dir, 'forecasting_results.csv')
    results_df.to_csv(results_path)
    print(f"Results saved: {results_path}")
    
    print("\n" + "="*80)
    print("FORECASTING TRAINING COMPLETE!")
    print("="*80)
    
    return results_df

if __name__ == "__main__":
    train_time_series_models()
