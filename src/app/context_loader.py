from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


def _context_path() -> Path:
    return Path(__file__).resolve().parents[2] / "context.json"


@lru_cache(maxsize=1)
def load_context() -> dict[str, Any]:
    path = _context_path()
    if not path.exists():
        raise FileNotFoundError(f"Missing context file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def context_summary() -> dict[str, Any]:
    ctx = load_context()
    return {
        "project_name": ctx.get("project_name"),
        "purpose": ctx.get("purpose"),
        "region": ctx.get("scope", {}).get("region"),
        "focus_corridors": ctx.get("scope", {}).get("focus_corridors", []),
        "deployment_target": ctx.get("scope", {}).get("deployment_target"),
        "current_models": ctx.get("models", {}).get("current_repo_models", []),
        "target_models": ctx.get("models", {}).get("target_models", []),
        "constraints": ctx.get("constraints", [])
    }
