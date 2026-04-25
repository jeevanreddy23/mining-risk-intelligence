from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from .config import DATASET_PATH


@dataclass
class DatasetConfig:
    rows: int = 2500
    seed: int = 42


def _clip(series: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.clip(series, low, high)


def _compute_risk_score(frame: pd.DataFrame) -> pd.Series:
    stress_ratio = frame["induced_stress_mpa"] / (frame["gsi"] * 0.9 + frame["rqd"] * 0.35 + 10.0)
    seismic_energy = np.power(10.0, 1.5 * frame["seismic_magnitude"])
    vibration = frame["ppv"] / 75.0
    structural = (
        (100.0 - frame["rqd"]) / 100.0 * 0.35
        + (100.0 - frame["gsi"]) / 100.0 * 0.35
        + (np.abs(frame["joint_angle_deg"] - 60.0) / 90.0) * 0.2
        + frame["groundwater"] * 0.1
    )
    seismic_component = np.clip(np.log10(seismic_energy + 1.0) / 3.0, 0.0, 1.0)
    score = (
        np.clip(stress_ratio, 0.0, 2.0) / 2.0 * 35.0
        + seismic_component * 25.0
        + np.clip(vibration, 0.0, 2.0) / 2.0 * 20.0
        + np.clip(structural, 0.0, 1.0) * 20.0
    )

    blast_penalty = np.where(
        (frame["charge_per_delay_kg"] > 28.0) & (frame["delay_interval_ms"] < 25.0),
        8.0,
        0.0,
    )
    return np.clip(score + blast_penalty, 0.0, 100.0)


def _label_hazard(row: pd.Series) -> str:
    if row["seismic_magnitude"] >= 1.8 and row["stress_ratio"] >= 1.1:
        return "Rockburst potential"
    if row["joint_angle_deg"] >= 45.0 and row["joint_angle_deg"] <= 75.0 and row["groundwater"] == 1 and row["gsi"] < 55.0:
        return "Wedge failure"
    if row["ppv"] >= 55.0 and row["charge_per_delay_kg"] >= 26.0:
        return "Blast-induced overbreak"
    if row["seismic_magnitude"] >= 1.2 and row["seismic_depth_m"] < 180.0:
        return "Seismic instability"
    return "Rockfall risk"


def _label_alert(score: float, seismic_magnitude: float) -> str:
    if score > 80.0 or seismic_magnitude >= 2.2:
        return "EVACUATE"
    if score >= 60.0:
        return "HIGH RISK"
    if score >= 30.0:
        return "CAUTION"
    return "SAFE"


def generate_synthetic_dataset(config: DatasetConfig | None = None) -> pd.DataFrame:
    config = config or DatasetConfig()
    rng = np.random.default_rng(config.seed)
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)

    timestamps = [start + timedelta(minutes=10 * i) for i in range(config.rows)]
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "ppv": _clip(rng.normal(28.0, 15.0, config.rows), 1.0, 110.0),
            "frequency_hz": _clip(rng.normal(28.0, 9.0, config.rows), 4.0, 80.0),
            "seismic_magnitude": _clip(rng.normal(0.8, 0.6, config.rows), -0.5, 3.0),
            "seismic_depth_m": _clip(rng.normal(220.0, 90.0, config.rows), 20.0, 650.0),
            "rqd": _clip(rng.normal(67.0, 18.0, config.rows), 20.0, 98.0),
            "gsi": _clip(rng.normal(58.0, 15.0, config.rows), 20.0, 90.0),
            "joint_angle_deg": _clip(rng.normal(54.0, 20.0, config.rows), 5.0, 88.0),
            "groundwater": rng.integers(0, 2, config.rows),
            "in_situ_stress_mpa": _clip(rng.normal(34.0, 8.0, config.rows), 10.0, 65.0),
            "charge_per_delay_kg": _clip(rng.normal(21.0, 7.0, config.rows), 3.0, 45.0),
            "delay_interval_ms": _clip(rng.normal(32.0, 10.0, config.rows), 8.0, 75.0),
            "burden_m": _clip(rng.normal(2.4, 0.4, config.rows), 1.2, 3.6),
            "spacing_m": _clip(rng.normal(2.6, 0.45, config.rows), 1.2, 4.0),
        }
    )
    frame["induced_stress_mpa"] = _clip(
        frame["in_situ_stress_mpa"] + rng.normal(10.0, 8.0, config.rows) + frame["seismic_magnitude"] * 4.0,
        12.0,
        90.0,
    )
    frame["stress_ratio"] = frame["induced_stress_mpa"] / (frame["gsi"] * 0.9 + frame["rqd"] * 0.35 + 10.0)
    frame["risk_score"] = _compute_risk_score(frame)
    frame["hazard_type"] = frame.apply(_label_hazard, axis=1)
    frame["alert_level"] = [
        _label_alert(score, mag)
        for score, mag in zip(frame["risk_score"], frame["seismic_magnitude"], strict=False)
    ]
    DATASET_PATH.parent.mkdir(exist_ok=True)
    frame.to_csv(DATASET_PATH, index=False)
    return frame
