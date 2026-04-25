from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "ppv",
    "frequency_hz",
    "seismic_magnitude",
    "seismic_depth_m",
    "rqd",
    "gsi",
    "joint_angle_deg",
    "groundwater",
    "in_situ_stress_mpa",
    "induced_stress_mpa",
    "charge_per_delay_kg",
    "delay_interval_ms",
    "burden_m",
    "spacing_m",
    "stress_strength_ratio",
    "joint_risk_index",
    "blast_intensity",
    "seismic_energy_index",
]


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    rock_strength_proxy = df["gsi"] * 0.9 + df["rqd"] * 0.35 + 10.0
    df["stress_strength_ratio"] = df["induced_stress_mpa"] / rock_strength_proxy
    df["joint_risk_index"] = (
        np.abs(df["joint_angle_deg"] - 60.0) / 60.0
        + (100.0 - df["gsi"]) / 100.0
        + df["groundwater"] * 0.25
    )
    df["blast_intensity"] = df["charge_per_delay_kg"] / np.maximum(df["delay_interval_ms"], 1.0)
    df["seismic_energy_index"] = np.log10(np.power(10.0, 1.5 * df["seismic_magnitude"]) + 1.0)
    return df[FEATURE_COLUMNS]
