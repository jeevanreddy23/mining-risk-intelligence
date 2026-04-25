from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight

from .config import DATA_DIR, METRICS_PATH, MODEL_PATH

FINAL_TABLE_PATH = DATA_DIR / 'final_training_table.csv'
CLASSIFICATION_REPORT_PATH = DATA_DIR / 'classification_report.json'
MODEL_COMPARISON_PATH = DATA_DIR / 'model_comparison.json'
TEST_PREDICTIONS_PATH = DATA_DIR / 'test_predictions.csv'
FEATURE_IMPORTANCE_PATH = DATA_DIR / 'feature_importance.csv'
FEATURE_CONTRACT_PATH = DATA_DIR / 'feature_contract.json'


RESEARCH_NOTES = {
    'random_forest': 'Breiman (2001) style tree ensemble logic for robust non-linear classification.',
    'hist_gradient_boosting': 'Gradient boosting family motivated by Friedman (2001); efficient edge-friendly tree boosting in scikit-learn.',
    'logistic_regression': 'Linear baseline for calibration and interpretability.',
}


def _load_training_table(dataset_path: Path) -> pd.DataFrame:
    if not dataset_path.exists():
        raise FileNotFoundError(
            f'Missing training table: {dataset_path}. Build data/final_training_table.csv first using the merge pipeline.'
        )
    return pd.read_csv(dataset_path)


def _split_columns(df: pd.DataFrame, target_col: str) -> tuple[list[str], list[str]]:
    feature_df = df.drop(columns=[target_col, 'label'], errors='ignore')
    feature_df = feature_df.dropna(axis=1, how='all')
    numeric_cols = feature_df.select_dtypes(include=['number', 'bool']).columns.tolist()
    categorical_cols = [col for col in feature_df.columns if col not in numeric_cols]
    return numeric_cols, categorical_cols


def _build_feature_contract(df: pd.DataFrame, feature_cols: list[str]) -> dict[str, Any]:
    contract: dict[str, Any] = {"feature_order": feature_cols, "defaults": {}, "dtypes": {}}
    for col in feature_cols:
        series = df[col]
        contract["dtypes"][col] = str(series.dtype)
        if pd.api.types.is_numeric_dtype(series):
            contract["defaults"][col] = float(series.dropna().median()) if series.notna().any() else 0.0
        else:
            mode = series.dropna().astype(str).mode()
            contract["defaults"][col] = str(mode.iloc[0]) if not mode.empty else ""
    return contract


def _build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='median')),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, numeric_cols),
            ('cat', categorical_pipeline, categorical_cols),
        ]
    )


def _build_models() -> dict[str, Any]:
    return {
        'logistic_regression': LogisticRegression(max_iter=2000, class_weight='balanced'),
        'random_forest': RandomForestClassifier(
            n_estimators=300,
            max_depth=16,
            min_samples_leaf=3,
            random_state=42,
            class_weight='balanced_subsample',
            n_jobs=-1,
        ),
        'hist_gradient_boosting': HistGradientBoostingClassifier(
            max_depth=10,
            learning_rate=0.08,
            max_iter=250,
            random_state=42,
        ),
    }


def _extract_feature_importance(pipeline: Pipeline) -> pd.DataFrame | None:
    model = pipeline.named_steps['model']
    preprocess = pipeline.named_steps['preprocess']
    if not hasattr(preprocess, 'get_feature_names_out'):
        return None
    if not hasattr(model, 'feature_importances_'):
        return None

    feature_names = preprocess.get_feature_names_out()
    importance = getattr(model, 'feature_importances_', None)
    if importance is None:
        return None

    df = pd.DataFrame({'feature': feature_names, 'importance': importance})
    return df.sort_values('importance', ascending=False)


def train_and_save_model(dataset_path: Path | None = None) -> dict[str, Any]:
    dataset_path = dataset_path or FINAL_TABLE_PATH
    dataset = _load_training_table(dataset_path)

    if 'target_label' not in dataset.columns:
        raise ValueError('Expected target_label column in final training table.')

    feature_frame = dataset.drop(columns=['target_label'])
    feature_frame = feature_frame.dropna(axis=1, how='all')
    x = feature_frame.copy()
    y = dataset['target_label'].astype(str)

    numeric_cols, categorical_cols = _split_columns(dataset, 'target_label')
    preprocessor = _build_preprocessor(numeric_cols, categorical_cols)

    class_counts = y.value_counts()
    stratify_target = y if int(class_counts.min()) >= 2 else None

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify_target,
    )

    sample_weight = compute_sample_weight(class_weight='balanced', y=y_train)

    comparison: dict[str, dict[str, Any]] = {}
    best_name = ''
    best_f1 = -1.0
    best_pipeline: Pipeline | None = None
    best_predictions: np.ndarray | None = None
    best_probabilities: np.ndarray | None = None

    for model_name, model in _build_models().items():
        pipeline = Pipeline(
            steps=[
                ('preprocess', preprocessor),
                ('model', model),
            ]
        )

        pipeline.fit(x_train, y_train, model__sample_weight=sample_weight)
        predictions = pipeline.predict(x_test)
        probabilities = pipeline.predict_proba(x_test) if hasattr(pipeline, 'predict_proba') else None
        weighted_f1 = f1_score(y_test, predictions, average='weighted')
        macro_f1 = f1_score(y_test, predictions, average='macro')
        accuracy = accuracy_score(y_test, predictions)

        comparison[model_name] = {
            'accuracy': float(accuracy),
            'weighted_f1': float(weighted_f1),
            'macro_f1': float(macro_f1),
            'research_note': RESEARCH_NOTES[model_name],
        }

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_name = model_name
            best_pipeline = pipeline
            best_predictions = predictions
            best_probabilities = probabilities

    assert best_pipeline is not None
    assert best_predictions is not None

    report = classification_report(y_test, best_predictions, output_dict=True)
    metrics = {
        'selected_model': best_name,
        'selection_metric': 'macro_f1',
        'selected_model_macro_f1': comparison[best_name]['macro_f1'],
        'selected_model_weighted_f1': comparison[best_name]['weighted_f1'],
        'selected_model_accuracy': comparison[best_name]['accuracy'],
        'dataset_path': str(dataset_path),
        'row_count': int(len(dataset)),
        'feature_count': int(x.shape[1]),
        'candidate_models': comparison,
        'stratified_split_used': stratify_target is not None,
    }

    prediction_df = x_test.reset_index(drop=True).copy()
    prediction_df['y_true'] = y_test.reset_index(drop=True)
    prediction_df['y_pred'] = pd.Series(best_predictions)
    if best_probabilities is not None:
        classes = best_pipeline.named_steps['model'].classes_
        for idx, class_name in enumerate(classes):
            prediction_df[f'prob_{class_name}'] = best_probabilities[:, idx]
        prediction_df['prediction_confidence'] = best_probabilities.max(axis=1)

    feature_importance_df = _extract_feature_importance(best_pipeline)
    feature_contract = _build_feature_contract(feature_frame, feature_frame.columns.tolist())

    joblib.dump(best_pipeline, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    CLASSIFICATION_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding='utf-8')
    MODEL_COMPARISON_PATH.write_text(json.dumps(comparison, indent=2), encoding='utf-8')
    FEATURE_CONTRACT_PATH.write_text(json.dumps(feature_contract, indent=2), encoding='utf-8')
    prediction_df.to_csv(TEST_PREDICTIONS_PATH, index=False)
    if feature_importance_df is not None:
        feature_importance_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
    return metrics
