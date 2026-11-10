"""Config loading and shared constants (MIMIC-IV item IDs, variable names)."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"

# ---------------------------------------------------------------------------
# MIMIC-IV item IDs (icu/d_items and hosp/d_labitems, MIMIC-IV v2.x / v3.x)
# ---------------------------------------------------------------------------
CHART_ITEMS: dict[str, list[int]] = {
    "heart_rate": [220045],
    "sbp": [220050, 220179],          # arterial / non-invasive systolic
    "dbp": [220051, 220180],
    "map": [220052, 220181, 225312],
    "resp_rate": [220210, 224690],
    "spo2": [220277],
    "temp_c": [223762],
    "temp_f": [223761],               # converted to Celsius
    "gcs_eye": [220739],
    "gcs_verbal": [223900],
    "gcs_motor": [223901],
    "fio2": [223835],
}

LAB_ITEMS: dict[str, list[int]] = {
    "lactate": [50813, 52442],
    "creatinine": [50912],
    "wbc": [51301, 51300],
    "hemoglobin": [51222],
    "platelets": [51265],
    "potassium": [50971],
    "sodium": [50983],
    "bicarbonate": [50882],
    "glucose": [50931],
    "bun": [51006],
}

VASOPRESSOR_ITEMS: dict[str, int] = {
    "norepinephrine": 221906,
    "epinephrine": 221289,
    "vasopressin": 222315,
    "phenylephrine": 221749,
    "dopamine": 221662,
}

# Plausible physiological ranges — values outside are treated as charting errors.
VALID_RANGES: dict[str, tuple[float, float]] = {
    "heart_rate": (20, 250), "sbp": (40, 260), "dbp": (15, 180), "map": (25, 200),
    "resp_rate": (3, 70), "spo2": (50, 100), "temp_c": (30, 43), "gcs_total": (3, 15),
    "fio2": (21, 100), "lactate": (0.2, 30), "creatinine": (0.1, 20), "wbc": (0.1, 200),
    "hemoglobin": (2, 25), "platelets": (2, 2000), "potassium": (1.5, 9), "sodium": (100, 180),
    "bicarbonate": (5, 55), "glucose": (20, 1500), "bun": (1, 250),
}

VITALS = ["heart_rate", "sbp", "dbp", "map", "resp_rate", "spo2", "temp_c", "gcs_total", "fio2"]
LABS = list(LAB_ITEMS)
DYNAMIC_VARS = VITALS + LABS


def _deep_update(base: dict, override: dict) -> dict:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict[str, Any]:
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if overrides:
        cfg = _deep_update(copy.deepcopy(cfg), overrides)
    # Resolve relative paths against the project root
    for key, val in cfg["paths"].items():
        p = Path(val)
        cfg["paths"][key] = str(p if p.is_absolute() else PROJECT_ROOT / p)
    return cfg


def forecast_horizons(cfg: dict) -> list[int]:
    """Hours ahead to forecast vitals for (default: every hour up to the label horizon)."""
    return list(cfg["forecast"].get("horizons") or range(1, cfg["labels"]["horizon_hours"] + 1))
