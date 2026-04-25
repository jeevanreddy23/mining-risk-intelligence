from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_ppv(distance_m: np.ndarray, charge_kg: np.ndarray, geology_factor: np.ndarray) -> np.ndarray:
    k = 1140.0
    n = 1.6
    scaled_distance = np.maximum(distance_m / np.maximum(np.sqrt(charge_kg), 1.0), 0.1)
    base = k * np.power(scaled_distance, -n)
    noise = np.random.normal(0.0, 0.12, size=len(distance_m)) * base
    return np.maximum(base * geology_factor + noise, 0.0)


def generate_synthetic_operational_data(rows: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    depth_m = rng.uniform(50, 900, rows)
    distance_to_structure_m = rng.uniform(5, 1500, rows)
    charge_per_delay_kg = rng.uniform(5, 60, rows)
    burden_m = rng.uniform(1.5, 4.5, rows)
    spacing_m = rng.uniform(1.8, 5.0, rows)
    delay_interval_ms = rng.uniform(8, 42, rows)
    dominant_frequency_hz = rng.uniform(5, 80, rows)
    structure_density = rng.uniform(0.0, 1.0, rows)
    groundwater = rng.integers(0, 2, rows)

    lithology_group = rng.choice(['mafic', 'felsic', 'ultramafic', 'sedimentary'], size=rows, p=[0.28, 0.32, 0.18, 0.22])
    geology_factor_map = {'mafic': 1.00, 'felsic': 0.92, 'ultramafic': 1.15, 'sedimentary': 0.85}
    geology_factor = np.array([geology_factor_map[item] for item in lithology_group])

    inferred_rqd_map = {'mafic': (55, 85), 'felsic': (65, 90), 'ultramafic': (30, 65), 'sedimentary': (35, 70)}
    inferred_gsi_map = {'mafic': (50, 75), 'felsic': (60, 80), 'ultramafic': (25, 55), 'sedimentary': (35, 60)}
    inferred_rqd = np.array([rng.uniform(*inferred_rqd_map[item]) for item in lithology_group])
    inferred_gsi = np.array([rng.uniform(*inferred_gsi_map[item]) for item in lithology_group])

    sigma_v_mpa = 0.027 * depth_m
    sigma_h_mpa = sigma_v_mpa * rng.uniform(1.2, 2.2, rows)

    ppv = generate_ppv(distance_to_structure_m + rng.uniform(10, 350, rows), charge_per_delay_kg, geology_factor)
    seismic_magnitude = np.clip(rng.normal(0.6 + structure_density * 0.8 + (900 - depth_m) / 3000, 0.45, rows), -0.5, 2.8)
    seismic_depth_m = np.clip(depth_m + rng.normal(0, 35, rows), 10, 1200)

    labels = np.full(rows, 'Normal', dtype=object)
    labels[(ppv > 100)] = 'Abnormal Blast Response'
    labels[(ppv > 50) & (inferred_rqd < 50) & (distance_to_structure_m < 250)] = 'Rockfall/Overbreak'
    labels[(seismic_magnitude > 1.6) & (sigma_h_mpa / np.maximum(sigma_v_mpa, 1.0) > 1.6)] = 'Seismic Instability'

    return pd.DataFrame(
        {
            'depth_m': depth_m,
            'lithology_group': lithology_group,
            'distance_to_structure_m': distance_to_structure_m,
            'structure_density': structure_density,
            'groundwater': groundwater,
            'charge_per_delay_kg': charge_per_delay_kg,
            'burden_m': burden_m,
            'spacing_m': spacing_m,
            'delay_interval_ms': delay_interval_ms,
            'dominant_frequency_hz': dominant_frequency_hz,
            'inferred_rqd': inferred_rqd,
            'inferred_gsi': inferred_gsi,
            'sigma_v_mpa': sigma_v_mpa,
            'sigma_h_mpa': sigma_h_mpa,
            'ppv_mm_s': ppv,
            'seismic_magnitude': seismic_magnitude,
            'seismic_depth_m': seismic_depth_m,
            'label': labels,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate synthetic mine-operational data for prototype ML training.')
    parser.add_argument('--output', required=True, help='Output CSV path.')
    parser.add_argument('--rows', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    df = generate_synthetic_operational_data(rows=args.rows, seed=args.seed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f'Wrote {len(df)} rows to {output_path}')


if __name__ == '__main__':
    main()
