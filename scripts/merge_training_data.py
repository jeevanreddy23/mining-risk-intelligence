from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PUBLIC_COLUMNS_PREFIX = 'public_'
SYNTHETIC_COLUMNS_PREFIX = 'synthetic_'


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f'Missing input file: {path}')
    return pd.read_csv(path)


def _prefix_columns(df: pd.DataFrame, prefix: str, skip: set[str] | None = None) -> pd.DataFrame:
    skip = skip or set()
    renamed = {}
    for column in df.columns:
        if column not in skip:
            renamed[column] = f'{prefix}{column}'
    return df.rename(columns=renamed)


def merge_training_data(
    public_features_path: Path,
    synthetic_operational_path: Path,
    output_path: Path,
    gravity_path: Path | None = None,
    seismic_path: Path | None = None,
) -> Path:
    public_df = _load_csv(public_features_path)
    synthetic_df = _load_csv(synthetic_operational_path)

    if gravity_path is not None and gravity_path.exists():
        gravity_df = _load_csv(gravity_path)
        if 'gravity_value' in gravity_df.columns and 'gravity_value' not in public_df.columns:
            public_df = public_df.copy()
            public_df['gravity_value'] = gravity_df['gravity_value']

    public_df = _prefix_columns(public_df, PUBLIC_COLUMNS_PREFIX)
    synthetic_df = _prefix_columns(synthetic_df, SYNTHETIC_COLUMNS_PREFIX, skip={'label'})

    row_count = min(len(public_df), len(synthetic_df))
    merged = pd.concat(
        [public_df.iloc[:row_count].reset_index(drop=True), synthetic_df.iloc[:row_count].reset_index(drop=True)],
        axis=1,
    )

    if seismic_path is not None and seismic_path.exists():
        seismic_df = _load_csv(seismic_path)
        if not seismic_df.empty:
            mag = pd.to_numeric(seismic_df.get('magnitude_value'), errors='coerce').dropna()
            depth = pd.to_numeric(seismic_df.get('depth_km'), errors='coerce').dropna()
            merged['public_regional_eq_count'] = len(seismic_df)
            merged['public_regional_eq_max_magnitude'] = float(mag.max()) if not mag.empty else np.nan
            merged['public_regional_eq_mean_depth_km'] = float(depth.mean()) if not depth.empty else np.nan

    merged['target_label'] = merged['label']
    merged['data_origin_note'] = 'public_context_plus_synthetic_operational_proxy'

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description='Merge public context features and synthetic operational data into one ML-ready table.')
    parser.add_argument('--public-features', required=True, help='CSV from feature engineering or gravity-enriched features.')
    parser.add_argument('--synthetic-operational', required=True, help='CSV from synthetic_operational_data.py.')
    parser.add_argument('--output', required=True, help='Output merged CSV path.')
    parser.add_argument('--gravity', help='Optional gravity-enriched CSV path.')
    parser.add_argument('--seismic', help='Optional GA seismic event CSV path.')
    args = parser.parse_args()

    output = merge_training_data(
        public_features_path=Path(args.public_features),
        synthetic_operational_path=Path(args.synthetic_operational),
        output_path=Path(args.output),
        gravity_path=Path(args.gravity) if args.gravity else None,
        seismic_path=Path(args.seismic) if args.seismic else None,
    )
    print(f'Wrote merged training data to {output}')


if __name__ == '__main__':
    main()
