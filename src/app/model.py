from __future__ import annotations

from dataclasses import dataclass

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class ModelBundle:
    pipeline: Pipeline


def build_model() -> ModelBundle:
    pipeline = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("model", GradientBoostingRegressor(random_state=42)),
        ]
    )
    return ModelBundle(pipeline=pipeline)
