"""Fit reproducible baseline models and save held-out evaluation artifacts."""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "medical_appointments.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


def evaluate_classifier():
    """Fit and evaluate a fixed-seed Random Forest on the saved split."""
    X_train = joblib.load(PROCESSED_DIR / "X_train.pkl")
    X_test = joblib.load(PROCESSED_DIR / "X_test.pkl")
    y_train = joblib.load(PROCESSED_DIR / "y_train.pkl")
    y_test = joblib.load(PROCESSED_DIR / "y_test.pkl")

    model = RandomForestClassifier(
        n_estimators=100,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])

    metrics = {
        "model": "RandomForestClassifier",
        "random_state": 42,
        "test_rows": int(len(y_test)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "confusion_matrix_labels": ["show", "no_show"],
        "confusion_matrix": matrix.tolist(),
    }

    fig, axis = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Show", "No-show"],
        yticklabels=["Show", "No-show"],
        ax=axis,
    )
    axis.set(title="No-Show Classifier: Held-Out Confusion Matrix", xlabel="Predicted", ylabel="Actual")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    importance = pd.Series(model.feature_importances_, index=X_train.columns)
    importance = importance.nlargest(15).sort_values()
    fig, axis = plt.subplots(figsize=(8, 6))
    importance.plot.barh(ax=axis, color="#2878a5")
    axis.set(title="Random Forest: Top 15 Feature Importances", xlabel="Impurity-based importance", ylabel="")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_importance.png", dpi=160)
    plt.close(fig)
    return metrics


def evaluate_demand_forecast():
    """Backtest daily appointment counts using a chronological 80/20 split."""
    appointments = pd.read_csv(RAW_DATA_PATH, usecols=["appointment_date_continuous"])
    appointments["appointment_date_continuous"] = pd.to_datetime(
        appointments["appointment_date_continuous"]
    )
    daily = (
        appointments.set_index("appointment_date_continuous")
        .resample("D")
        .size()
        .rename("appointments")
        .to_frame()
    )
    daily["day_of_week"] = daily.index.dayofweek
    daily["month"] = daily.index.month
    daily["day_of_year"] = daily.index.dayofyear

    for lag in (1, 7, 14, 28):
        daily[f"lag_{lag}"] = daily["appointments"].shift(lag)
    for window in (7, 28):
        daily[f"rolling_mean_{window}"] = (
            daily["appointments"].shift(1).rolling(window).mean()
        )

    daily = daily.dropna()
    split_at = int(0.8 * len(daily))
    train = daily.iloc[:split_at]
    test = daily.iloc[split_at:]
    feature_columns = [column for column in daily.columns if column != "appointments"]

    model = HistGradientBoostingRegressor(max_iter=100, random_state=42)
    model.fit(train[feature_columns], train["appointments"])
    predictions = np.maximum(0, model.predict(test[feature_columns]))
    actual = test["appointments"].to_numpy()
    nonzero = actual != 0
    mape = (
        float(np.mean(np.abs((actual[nonzero] - predictions[nonzero]) / actual[nonzero])) * 100)
        if nonzero.any()
        else None
    )

    metrics = {
        "model": "HistGradientBoostingRegressor",
        "random_state": 42,
        "split": "chronological 80/20",
        "test_days": int(len(test)),
        "mape_percent_nonzero_days": mape,
        "mae_appointments_per_day": float(mean_absolute_error(actual, predictions)),
        "rmse_appointments_per_day": float(np.sqrt(mean_squared_error(actual, predictions))),
        "r2": float(r2_score(actual, predictions)),
    }

    forecast = pd.DataFrame(
        {"date": test.index, "actual_appointments": actual, "predicted_appointments": predictions}
    )
    forecast.to_csv(REPORTS_DIR / "demand_forecast_backtest.csv", index=False)
    fig, axis = plt.subplots(figsize=(11, 5))
    axis.plot(forecast["date"], forecast["actual_appointments"], label="Actual", linewidth=1.5)
    axis.plot(forecast["date"], forecast["predicted_appointments"], label="Predicted", linewidth=1.5)
    axis.set(title="Daily Demand: Chronological Held-Out Backtest", xlabel="Date", ylabel="Appointments")
    axis.legend()
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "demand_forecast_backtest.png", dpi=160)
    plt.close(fig)
    return metrics


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    metrics = {
        "classification": evaluate_classifier(),
        "demand_forecasting": evaluate_demand_forecast(),
    }
    with (REPORTS_DIR / "metrics.json").open("w", encoding="utf-8") as output:
        json.dump(metrics, output, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()