from __future__ import annotations

import joblib
import pandas as pd

from .config import MODEL_PATH


ALERT_MAP = {
    'Normal': 'SAFE',
    'Abnormal Blast Response': 'HIGH RISK',
    'Rockfall/Overbreak': 'HIGH RISK',
    'Seismic Instability': 'EVACUATE',
}


def load_model():
    if not MODEL_PATH.exists():
        from .training import train_and_save_model

        train_and_save_model()
    return joblib.load(MODEL_PATH)


def _confidence_label(confidence: float) -> str:
    if confidence >= 0.8:
        return 'high'
    if confidence >= 0.6:
        return 'medium'
    return 'low'


def _top_probability_drivers(probabilities: dict[str, float]) -> list[str]:
    return [name for name, _ in sorted(probabilities.items(), key=lambda item: item[1], reverse=True)[:3]]


def _build_failure_mechanism(predicted_label: str, packet: dict[str, float]) -> str:
    if predicted_label == 'Rockfall/Overbreak':
        return 'High PPV combined with lower inferred rock quality and structure proximity suggests local wall or back damage potential.'
    if predicted_label == 'Abnormal Blast Response':
        return 'The blast-response pattern indicates elevated vibration severity relative to structural and geometric conditions.'
    if predicted_label == 'Seismic Instability':
        return 'Seismic magnitude and stress proxy features indicate elevated instability risk requiring conservative controls.'
    return 'Current feature combination is most consistent with normal prototype operating conditions.'


def score_packet(packet: dict[str, float]) -> dict[str, object]:
    model = load_model()
    row = pd.DataFrame([packet])

    predicted_label = str(model.predict(row)[0])
    probabilities = {}
    confidence = 0.0
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(row)[0]
        classes = model.named_steps['model'].classes_
        probabilities = {str(label): float(prob) for label, prob in zip(classes, probs, strict=False)}
        confidence = max(probabilities.values()) if probabilities else 0.0

    risk_score = round(confidence * 100)
    alert_level = ALERT_MAP.get(predicted_label, 'CAUTION')

    return {
        'Hazard Type': predicted_label,
        'Risk Score': risk_score,
        'Alert Level': alert_level,
        'Failure Mechanism': _build_failure_mechanism(predicted_label, packet),
        'Key Indicators': {
            'PPV': f"{packet['synthetic_ppv_mm_s']:.1f} mm/s",
            'Seismic Energy': f"M{packet['synthetic_seismic_magnitude']:.2f} at {packet['synthetic_seismic_depth_m']:.0f} m",
            'Rock Mass Condition': f"RQD {packet['synthetic_inferred_rqd']:.0f}, GSI {packet['synthetic_inferred_gsi']:.0f}, structure distance {packet['synthetic_distance_to_structure_m']:.0f} m",
        },
        'Blast Insight (if relevant)': f"Charge per delay {packet['synthetic_charge_per_delay_kg']:.1f} kg, burden {packet['synthetic_burden_m']:.2f} m, spacing {packet['synthetic_spacing_m']:.2f} m.",
        'Confidence Level': _confidence_label(confidence),
        'Class Probabilities': probabilities,
        'Top Drivers': _top_probability_drivers(probabilities) if probabilities else [],
    }
