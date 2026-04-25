from __future__ import annotations

from fastapi import FastAPI

from .inference import score_packet
from .schemas import SensorPacket


app = FastAPI(title="Edge Mining Risk Pipeline", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/score")
def score(sensor_packet: SensorPacket) -> dict[str, object]:
    return score_packet(sensor_packet.model_dump())
