from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class SensorPacket(BaseModel):
    timestamp: datetime

    public_distance_to_mineral_m: float = Field(default=500.0, ge=0.0)
    public_drillhole_count_1km: float = Field(default=0.0, ge=0.0)
    public_drillhole_count_5km: float = Field(default=0.0, ge=0.0)
    public_gravity_value: float = Field(default=0.0)
    public_regional_eq_count: float = Field(default=0.0, ge=0.0)
    public_regional_eq_max_magnitude: float = Field(default=0.0)
    public_regional_eq_mean_depth_km: float = Field(default=0.0, ge=0.0)

    synthetic_depth_m: float = Field(ge=0.0, le=3000.0)
    synthetic_lithology_group: str = Field(default='felsic')
    synthetic_distance_to_structure_m: float = Field(ge=0.0, le=5000.0)
    synthetic_structure_density: float = Field(ge=0.0, le=1.0)
    synthetic_groundwater: int = Field(ge=0, le=1)
    synthetic_charge_per_delay_kg: float = Field(ge=0.0, le=1000.0)
    synthetic_burden_m: float = Field(ge=0.1, le=20.0)
    synthetic_spacing_m: float = Field(ge=0.1, le=20.0)
    synthetic_delay_interval_ms: float = Field(ge=1.0, le=250.0)
    synthetic_dominant_frequency_hz: float = Field(ge=0.0, le=500.0)
    synthetic_inferred_rqd: float = Field(ge=0.0, le=100.0)
    synthetic_inferred_gsi: float = Field(ge=0.0, le=100.0)
    synthetic_sigma_v_mpa: float = Field(ge=0.0, le=200.0)
    synthetic_sigma_h_mpa: float = Field(ge=0.0, le=300.0)
    synthetic_ppv_mm_s: float = Field(ge=0.0, le=500.0)
    synthetic_seismic_magnitude: float = Field(ge=-2.0, le=5.0)
    synthetic_seismic_depth_m: float = Field(ge=0.0, le=3000.0)
