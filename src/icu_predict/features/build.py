"""Leak-free hourly features.

Every feature for hour t is computed only from rows with hour <= t, using
per-stay forward fills, backward-looking rolling windows and shifts. The same
function powers training, the API and the dashboard, so there is a single
definition of each feature.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from icu_predict.config import LABS, VITALS

ROLL_VARS = ["heart_rate", "sbp", "map", "resp_rate", "spo2", "temp_c"]
TREND_LABS = ["lactate", "creatinine", "platelets", "bicarbonate"]
CAREUNIT_KEYS = {"micu": "MICU", "sicu": "SICU", "cvicu": "CVICU", "ccu": "CCU", "tsicu": "TSICU", "neuro": "Neuro"}
STATIC_COLS = ["age", "male", "emergency_admission"] + [f"unit_{k}" for k in CAREUNIT_KEYS]


def static_features(cohort: pd.DataFrame) -> pd.DataFrame:
    s = cohort[["stay_id", "age", "male", "emergency_admission"]].copy()
    unit = cohort.get("first_careunit", pd.Series([""] * len(cohort), index=cohort.index)).fillna("")
    for key, token in CAREUNIT_KEYS.items():
        if key == "sicu":   # avoid matching TSICU and MICU/SICU twice in a confusing way
            s[f"unit_{key}"] = unit.str.contains(r"\(SICU\)|MICU/SICU").astype(int)
        elif key == "micu":
            s[f"unit_{key}"] = unit.str.contains(r"\(MICU\)|MICU/SICU").astype(int)
        else:
            s[f"unit_{key}"] = unit.str.contains(token, regex=False).astype(int)
    return s


def news2_score(rr, spo2, temp, sbp, hr, gcs) -> np.ndarray:
    """Simplified NEWS2 (no supplemental-O2 term); NaN components score 0."""
    def band(x, edges, scores):
        x = np.asarray(x, dtype=float)
        out = np.zeros_like(x)
        for (lo, hi), sc in zip(edges, scores):
            out[(x >= lo) & (x < hi)] = sc
        return out
    total = (band(rr, [(-1, 9), (9, 12), (21, 25), (25, 999)], [3, 1, 2, 3])
             + band(spo2, [(-1, 92), (92, 94), (94, 96)], [3, 2, 1])
             + band(temp, [(-1, 35.05), (35.05, 36.05), (38.05, 39.05), (39.05, 99)], [3, 1, 1, 2])
             + band(sbp, [(-1, 91), (91, 101), (101, 111), (220, 999)], [3, 2, 1, 3])
             + band(hr, [(-1, 41), (41, 51), (91, 111), (111, 131), (131, 999)], [3, 1, 1, 2, 3])
             + band(gcs, [(-1, 15)], [3]))
    return total


def build_features(timeline: pd.DataFrame, cohort: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    tl = timeline.sort_values(["stay_id", "hour"]).reset_index(drop=True).copy()
    for v in VITALS + LABS:
        tl[v] = pd.to_numeric(tl[v], errors="coerce").astype(float) if v in tl.columns else np.nan
    if "vasopressor" not in tl.columns:
        tl["vasopressor"] = 0
    # Fill MAP from SBP/DBP when MAP itself wasn't charted
    calc_map = tl["dbp"] + (tl["sbp"] - tl["dbp"]) / 3
    tl["map"] = tl["map"].fillna(calc_map)

    g = tl.groupby("stay_id", sort=False)
    feats = {"stay_id": tl["stay_id"].to_numpy(), "hour": tl["hour"].to_numpy()}
    lim_v, lim_l = cfg["timeline"]["ffill_limit_vitals"], cfg["timeline"]["ffill_limit_labs"]

    last = {}
    for v in VITALS + LABS:
        lim = lim_v if v in VITALS else lim_l
        last[v] = g[v].ffill(limit=lim)
        feats[f"{v}_last"] = last[v].to_numpy()
        # hours since last measurement (capped at 72; 72 = never measured)
        idx = tl["hour"].where(tl[v].notna())
        since = tl["hour"] - idx.groupby(tl["stay_id"]).ffill()
        feats[f"{v}_hrs_since"] = since.fillna(72).clip(upper=72).to_numpy()

    for v in ROLL_VARS:
        r = g[v].rolling(6, min_periods=1)
        feats[f"{v}_mean6"] = r.mean().reset_index(level=0, drop=True).to_numpy()
        feats[f"{v}_min6"] = r.min().reset_index(level=0, drop=True).to_numpy()
        feats[f"{v}_max6"] = r.max().reset_index(level=0, drop=True).to_numpy()
        feats[f"{v}_std6"] = r.std().reset_index(level=0, drop=True).to_numpy()
        feats[f"{v}_delta1"] = (last[v] - last[v].groupby(tl["stay_id"]).shift(1)).to_numpy()
        feats[f"{v}_delta6"] = (last[v] - last[v].groupby(tl["stay_id"]).shift(6)).to_numpy()

    for v in TREND_LABS:  # change since the previous distinct measurement
        meas = tl[["stay_id", v]].dropna()
        prev = meas.groupby("stay_id")[v].shift(1)
        d = pd.Series(np.nan, index=tl.index)
        d.loc[meas.index] = meas[v] - prev
        feats[f"{v}_change"] = d.groupby(tl["stay_id"]).ffill(limit=lim_l).to_numpy()

    feats["shock_index"] = (last["heart_rate"] / last["sbp"]).to_numpy()
    feats["pulse_pressure"] = (last["sbp"] - last["dbp"]).to_numpy()
    feats["news2"] = news2_score(last["resp_rate"], last["spo2"], last["temp_c"], last["sbp"],
                                 last["heart_rate"], last["gcs_total"])
    feats["vasopressor_now"] = tl["vasopressor"].to_numpy()
    feats["vasopressor_hours"] = g["vasopressor"].cumsum().to_numpy()
    feats["hours_in_icu"] = (tl["hour"] + 1).to_numpy()

    f = pd.DataFrame(feats)
    f = f.merge(static_features(cohort), on="stay_id", how="left")
    return f


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("stay_id", "hour")]
