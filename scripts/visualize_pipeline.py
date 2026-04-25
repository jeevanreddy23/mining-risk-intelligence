from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


sns.set_theme(style='whitegrid')


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_label_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    df['target_label'].value_counts().plot(kind='bar', ax=ax, color='#2c7fb8')
    ax.set_title('Training Label Distribution')
    ax.set_xlabel('Label')
    ax.set_ylabel('Count')
    fig.tight_layout()
    fig.savefig(output_dir / 'label_distribution.png', dpi=200)
    plt.close(fig)


def plot_numeric_correlation(df: pd.DataFrame, output_dir: Path) -> None:
    numeric = df.select_dtypes(include=['number']).copy()
    keep = [col for col in numeric.columns if col in {
        'public_distance_to_mineral_m',
        'public_drillhole_count_1km',
        'public_gravity_value',
        'synthetic_ppv_mm_s',
        'synthetic_charge_per_delay_kg',
        'synthetic_inferred_rqd',
        'synthetic_inferred_gsi',
        'synthetic_sigma_h_mpa',
        'synthetic_seismic_magnitude',
    }]
    if len(keep) < 2:
        keep = numeric.columns[:10].tolist()
    corr = numeric[keep].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap='coolwarm', center=0, ax=ax)
    ax.set_title('Numeric Feature Correlation Heatmap')
    fig.tight_layout()
    fig.savefig(output_dir / 'correlation_heatmap.png', dpi=200)
    plt.close(fig)


def plot_key_feature_distributions(df: pd.DataFrame, output_dir: Path) -> None:
    candidates = [
        'public_distance_to_mineral_m',
        'public_gravity_value',
        'synthetic_ppv_mm_s',
        'synthetic_inferred_rqd',
        'synthetic_seismic_magnitude',
    ]
    columns = [col for col in candidates if col in df.columns]
    if not columns:
        return
    fig, axes = plt.subplots(len(columns), 1, figsize=(9, 3 * len(columns)))
    if len(columns) == 1:
        axes = [axes]
    for ax, col in zip(axes, columns, strict=False):
        sns.histplot(df[col].dropna(), kde=True, ax=ax, color='#41ab5d')
        ax.set_title(f'Distribution: {col}')
    fig.tight_layout()
    fig.savefig(output_dir / 'feature_distributions.png', dpi=200)
    plt.close(fig)


def plot_confusion(pred_df: pd.DataFrame, output_dir: Path) -> None:
    labels = sorted(set(pred_df['y_true']).union(set(pred_df['y_pred'])))
    cm = confusion_matrix(pred_df['y_true'], pred_df['y_pred'], labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title('Confusion Matrix')
    fig.tight_layout()
    fig.savefig(output_dir / 'confusion_matrix.png', dpi=200)
    plt.close(fig)


def plot_prediction_confidence(pred_df: pd.DataFrame, output_dir: Path) -> None:
    if 'prediction_confidence' not in pred_df.columns:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(pred_df['prediction_confidence'].dropna(), bins=20, kde=True, ax=ax, color='#756bb1')
    ax.set_title('Prediction Confidence Distribution')
    ax.set_xlabel('Confidence')
    fig.tight_layout()
    fig.savefig(output_dir / 'prediction_confidence.png', dpi=200)
    plt.close(fig)


def plot_test_label_comparison(pred_df: pd.DataFrame, output_dir: Path) -> None:
    compare = pd.DataFrame({
        'True': pred_df['y_true'].value_counts(),
        'Predicted': pred_df['y_pred'].value_counts(),
    }).fillna(0)
    fig, ax = plt.subplots(figsize=(9, 5))
    compare.plot(kind='bar', ax=ax)
    ax.set_title('Test Set: True vs Predicted Label Counts')
    ax.set_ylabel('Count')
    fig.tight_layout()
    fig.savefig(output_dir / 'test_label_comparison.png', dpi=200)
    plt.close(fig)


def plot_feature_importance(feature_df: pd.DataFrame, output_dir: Path) -> None:
    top = feature_df.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top['feature'], top['importance'], color='#dd8452')
    ax.set_title('Top Feature Importances')
    ax.set_xlabel('Importance')
    fig.tight_layout()
    fig.savefig(output_dir / 'feature_importance.png', dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate pre-training and post-training visualisations for the mining ML pipeline.')
    parser.add_argument('--training-table', required=True, help='Path to final training table CSV.')
    parser.add_argument('--predictions', help='Path to test_predictions.csv from training output.')
    parser.add_argument('--feature-importance', help='Optional path to feature_importance.csv.')
    parser.add_argument('--output-dir', required=True, help='Directory to save plots.')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    _ensure_dir(output_dir)

    training_df = pd.read_csv(args.training_table)
    plot_label_distribution(training_df, output_dir)
    plot_numeric_correlation(training_df, output_dir)
    plot_key_feature_distributions(training_df, output_dir)

    if args.predictions and Path(args.predictions).exists():
        pred_df = pd.read_csv(args.predictions)
        plot_confusion(pred_df, output_dir)
        plot_prediction_confidence(pred_df, output_dir)
        plot_test_label_comparison(pred_df, output_dir)

    if args.feature_importance and Path(args.feature_importance).exists():
        feature_df = pd.read_csv(args.feature_importance)
        if not feature_df.empty:
            plot_feature_importance(feature_df, output_dir)


if __name__ == '__main__':
    main()
