from __future__ import annotations

from typing import Any


def hazard_from_inputs(data: dict[str, float]) -> str:
    if data["seismic_magnitude"] >= 1.8 and data["stress_strength_ratio"] >= 1.1:
        return "Rockburst potential"
    if 45.0 <= data["joint_angle_deg"] <= 75.0 and data["groundwater"] >= 1 and data["gsi"] < 55.0:
        return "Wedge failure"
    if data["ppv"] >= 55.0 and data["charge_per_delay_kg"] >= 26.0:
        return "Blast-induced overbreak"
    if data["seismic_magnitude"] >= 1.2 and data["seismic_depth_m"] < 180.0:
        return "Seismic instability"
    return "Rockfall risk"


def alert_from_score(score: float, seismic_magnitude: float) -> str:
    if score > 80.0 or seismic_magnitude >= 2.2:
        return "EVACUATE"
    if score >= 60.0:
        return "HIGH RISK"
    if score >= 30.0:
        return "CAUTION"
    return "SAFE"


def apply_rule_overrides(base_score: float, data: dict[str, Any]) -> tuple[float, list[str]]:
    score = base_score
    reasons: list[str] = []

    if data["ppv"] > 65.0 and data["gsi"] < 45.0:
        score = max(score, 78.0)
        reasons.append("High vibration with poor rock mass")
    if data["seismic_magnitude"] > 1.8 and data["seismic_depth_m"] < 150.0:
        score = max(score, 85.0)
        reasons.append("Shallow significant seismic event")
    if data["stress_strength_ratio"] > 1.15:
        score = max(score, 74.0)
        reasons.append("Stress exceeds conservative strength proxy")
    if 45.0 <= data["joint_angle_deg"] <= 75.0 and data["groundwater"] >= 1 and data["ppv"] > 35.0:
        score = max(score, 70.0)
        reasons.append("Joint-water-vibration wedge condition")
    if data["charge_per_delay_kg"] > 30.0 and data["delay_interval_ms"] < 25.0 and data["ppv"] > 50.0:
        score = max(score, 68.0)
        reasons.append("Blast design likely driving overbreak")

    return min(score, 100.0), reasons


def blast_recommendation(data: dict[str, Any]) -> str:
    if data["charge_per_delay_kg"] > 30.0 or data["ppv"] > 50.0:
        return "Reduce charge per delay, increase delay intervals, and review burden/spacing near critical excavations."
    if data["blast_intensity"] > 1.0:
        return "Review delay scatter and charge concentration; consider decoupled charges to limit wall damage."
    return "No strong blast design concern detected from current inputs."


def failure_mechanism(hazard_type: str, reasons: list[str]) -> str:
    base = {
        "Rockburst potential": "Elevated stress concentration and seismic loading may trigger violent strain release.",
        "Wedge failure": "Adverse joint orientation with water and vibration may daylight unstable wedges.",
        "Blast-induced overbreak": "High blast energy transfer is likely damaging excavation walls beyond design.",
        "Seismic instability": "Shallow seismic activity suggests local dynamic instability around the excavation.",
        "Rockfall risk": "Degraded rock mass and vibration may detach blocks from backs or walls.",
    }[hazard_type]
    if reasons:
        return f"{base} Triggered by: {', '.join(reasons[:2])}."
    return base
