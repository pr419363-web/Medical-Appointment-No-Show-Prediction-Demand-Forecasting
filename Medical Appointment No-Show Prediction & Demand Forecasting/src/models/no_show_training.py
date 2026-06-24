import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix, classification_report

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None


def load_processed(path='data/processed_medical_appointments.csv'):
    p = Path(path)
    if not p.exists():
        # try parent path
        p = Path('..') / path
        if not p.exists():
            raise FileNotFoundError(f"Processed CSV not found at {path}")
    df = pd.read_csv(p, low_memory=False)
    return df


def prepare_data(df, target='no_show'):
    df = df.copy()
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataframe")

    y = df[target].astype(int)
    # drop identifiers and raw date/time columns
    drop_cols = [c for c in ['appointment_date', 'appointment_time', 'appointment_date_continuous'] if c in df.columns]
    X = df.drop(columns=[target] + drop_cols, errors='ignore')

    # identify numeric and categorical
    numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = [c for c in X.select_dtypes(include=['object', 'category']) .columns.tolist()]

    return X, y, numeric_cols, categorical_cols


def build_preprocessor(numeric_cols, categorical_cols):
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('ohe', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ], remainder='drop')

    return preprocessor


def train_and_evaluate(df, target='no_show', test_size=0.2, random_state=42, save_dir='models'):
    X, y, numeric_cols, categorical_cols = prepare_data(df, target=target)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    classifiers = {
        'logistic': LogisticRegression(class_weight='balanced', max_iter=2000, random_state=random_state),
        'random_forest': RandomForestClassifier(n_estimators=200, class_weight='balanced', n_jobs=-1, random_state=random_state)
    }
    if XGBClassifier is not None:
        classifiers['xgboost'] = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state)

    results = []

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    for name, clf in classifiers.items():
        pipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        try:
            y_proba = pipe.predict_proba(X_test)[:, 1]
        except Exception:
            y_proba = None

        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        results.append({
            'model': name,
            'f1': f1,
            'roc_auc': roc,
            'precision': prec,
            'recall': rec,
            'confusion_matrix': cm
        })

        model_file = save_path / f'no_show_{name}.joblib'
        joblib.dump(pipe, model_file)

    res_df = pd.DataFrame(results).sort_values(by='f1', ascending=False).reset_index(drop=True)

    # Save best model
    best = res_df.iloc[0]['model']
    best_file = save_path / f'no_show_{best}.joblib'
    if best_file.exists():
        joblib.copy(best_file, save_path / 'no_show_best.joblib')
    else:
        # fallback: save the pipeline for the best entry
        best_pipe = None
        for name, clf in classifiers.items():
            if name == best:
                best_pipe = Pipeline(steps=[('preprocessor', build_preprocessor(numeric_cols, categorical_cols)), ('classifier', classifiers[name])])
                # retrain on full data
                best_pipe.fit(X, y)
                joblib.dump(best_pipe, save_path / 'no_show_best.joblib')
                break

    return res_df


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed', default='../data/processed_medical_appointments.csv')
    parser.add_argument('--out', default='../models')
    args = parser.parse_args()

    df = load_processed(args.processed)
    res = train_and_evaluate(df, save_dir=args.out)
    print(res)
