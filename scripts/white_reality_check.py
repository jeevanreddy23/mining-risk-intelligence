from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


def _split_columns(df: pd.DataFrame, target_col: str) -> tuple[list[str], list[str], pd.DataFrame]:
    feature_df = df.drop(columns=[target_col, "label"], errors="ignore")
    feature_df = feature_df.dropna(axis=1, how="all")
    numeric_cols = feature_df.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [col for col in feature_df.columns if col not in numeric_cols]
    return numeric_cols, categorical_cols, feature_df


def _build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]), numeric_cols),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )


def _build_models() -> dict[str, Any]:
    return {
        "logistic_regression": LogisticRegression(max_iter=3000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=16,
            min_samples_leaf=3,
            random_state=42,
            class_weight="balanced_subsample",
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_depth=10,
            learning_rate=0.08,
            max_iter=250,
            random_state=42,
        ),
    }


def _build_splitter(y: pd.Series, n_splits: int, random_state: int) -> Any:
    if int(y.value_counts().min()) >= n_splits:
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def _evaluate_models(dataset: pd.DataFrame, target_col: str, n_splits: int, random_state: int) -> tuple[pd.DataFrame, dict[str, list[float]]]:
    y = dataset[target_col].astype(str)
    numeric_cols, categorical_cols, x = _split_columns(dataset, target_col)
    splitter = _build_splitter(y, n_splits=n_splits, random_state=random_state)
    preprocessor = _build_preprocessor(numeric_cols, categorical_cols)

    rows: list[dict[str, Any]] = []
    fold_losses: dict[str, list[float]] = {name: [] for name in _build_models()}

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(x, y), start=1):
        x_train = x.iloc[train_idx]
        x_test = x.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        for model_name, model in _build_models().items():
            pipeline = Pipeline(
                steps=[
                    ("preprocess", preprocessor),
                    ("model", model),
                ]
            )
            pipeline.fit(x_train, y_train)
            predictions = pipeline.predict(x_test)
            accuracy = accuracy_score(y_test, predictions)
            macro_f1 = f1_score(y_test, predictions, average="macro")
            loss = 1.0 - macro_f1

            rows.append(
                {
                    "fold": fold_idx,
                    "model": model_name,
                    "accuracy": float(accuracy),
                    "macro_f1": float(macro_f1),
                    "loss": float(loss),
                }
            )
            fold_losses[model_name].append(float(loss))

    return pd.DataFrame(rows), fold_losses


def _white_reality_check(
    fold_losses: dict[str, list[float]],
    benchmark: str,
    n_bootstrap: int,
    random_state: int,
) -> dict[str, Any]:
    if benchmark not in fold_losses:
        raise ValueError(f"Benchmark model '{benchmark}' not found.")

    rng = np.random.default_rng(random_state)
    benchmark_losses = np.array(fold_losses[benchmark], dtype=float)
    candidate_names = [name for name in fold_losses if name != benchmark]

    if not candidate_names:
        raise ValueError("Need at least one candidate model beyond the benchmark.")

    differential_matrix = []
    observed_means = {}
    for name in candidate_names:
        diff = benchmark_losses - np.array(fold_losses[name], dtype=float)
        differential_matrix.append(diff)
        observed_means[name] = float(diff.mean())

    differentials = np.vstack(differential_matrix)
    observed_stat = float(np.max(differentials.mean(axis=1)))
    centered = differentials - differentials.mean(axis=1, keepdims=True)

    bootstrap_stats = []
    n_obs = centered.shape[1]
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n_obs, n_obs)
        sampled = centered[:, idx]
        bootstrap_stats.append(float(np.max(sampled.mean(axis=1))))

    bootstrap_array = np.array(bootstrap_stats)
    p_value = float(np.mean(bootstrap_array >= observed_stat))

    return {
        "benchmark_model": benchmark,
        "candidate_models": candidate_names,
        "observed_mean_differentials": observed_means,
        "observed_test_statistic": observed_stat,
        "bootstrap_p_value": p_value,
        "bootstrap_iterations": n_bootstrap,
        "interpretation": (
            "Low p-value suggests the best candidate outperforms the benchmark beyond data-snooping effects."
            if p_value < 0.1
            else "No strong evidence that the selected candidate beats the benchmark after data-snooping adjustment."
        ),
    }


def run_white_reality_check(
    input_path: Path,
    output_dir: Path,
    target_col: str = "target_label",
    benchmark: str = "logistic_regression",
    n_splits: int = 5,
    n_bootstrap: int = 2000,
    random_state: int = 42,
) -> dict[str, Any]:
    dataset = pd.read_csv(input_path)
    if target_col not in dataset.columns:
        raise ValueError(f"Target column '{target_col}' not found in {input_path}.")

    output_dir.mkdir(parents=True, exist_ok=True)

    fold_metrics, fold_losses = _evaluate_models(
        dataset=dataset,
        target_col=target_col,
        n_splits=n_splits,
        random_state=random_state,
    )
    fold_metrics.to_csv(output_dir / "white_reality_check_fold_metrics.csv", index=False)

    summary = _white_reality_check(
        fold_losses=fold_losses,
        benchmark=benchmark,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    )
    (output_dir / "white_reality_check.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report = f"""# White Reality Check

## Purpose
This test checks whether the best-performing model appears to beat the benchmark after adjusting for model selection / data snooping.

## Settings
- benchmark: `{benchmark}`
- target column: `{target_col}`
- CV folds: {n_splits}
- bootstrap iterations: {n_bootstrap}

## Result
- observed statistic: {summary['observed_test_statistic']:.6f}
- bootstrap p-value: {summary['bootstrap_p_value']:.6f}

## Interpretation
{summary['interpretation']}

## Caution
- This is still a prototype validation layer.
- The final training table includes synthetic proxy labels and synthetic operational features.
- A strong result here does not replace site validation or mine-grade holdout testing.
"""
    (output_dir / "white_reality_check_report.md").write_text(report, encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run White's Reality Check on candidate mining-risk models to reduce data-snooping / overfitting risk."
    )
    parser.add_argument("--input", required=True, help="Path to final training table CSV.")
    parser.add_argument("--output", required=True, help="Directory for White Reality Check outputs.")
    parser.add_argument("--target-col", default="target_label")
    parser.add_argument("--benchmark", default="logistic_regression")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    summary = run_white_reality_check(
        input_path=Path(args.input),
        output_dir=Path(args.output),
        target_col=args.target_col,
        benchmark=args.benchmark,
        n_splits=args.folds,
        n_bootstrap=args.bootstrap,
        random_state=args.seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
