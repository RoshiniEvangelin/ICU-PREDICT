"""Six-hour deterioration labels and vital-sign forecast targets.

A *deterioration event* starts at hour h if any of these happen at h:
  * vasopressor started (first infusion of norepinephrine / epinephrine /
    vasopressin / phenylephrine / dopamine)
  * death in the ICU
  * sustained instability: a threshold breach (SBP<90, MAP<65, HR>130,
    RR>30, SpO2<88, lactate>4) in `sustained_hours` consecutive hours,
    with h the hour that completes the run

The prediction at hour t (made at the end of hour t) is labelled 1 if an event
starts in hours t+1 ... t+horizon, else 0. Only "currently stable" hours are
eligible — not unstable at t, not on vasopressors, and (by default) before the
stay's first event — because predicting that an already-crashing patient is
crashing is trivially easy and inflates metrics. Negatives also need vitals
recorded in enough of the future window, so "no data" isn't read as "no event".
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def unstable_flags(tl: pd.DataFrame, th: dict) -> pd.Series:
    sbp_map = tl["map"].fillna(tl["dbp"] + (tl["sbp"] - tl["dbp"]) / 3)
    return ((tl["sbp"] < th["sbp_low"]) | (sbp_map < th["map_low"]) | (tl["heart_rate"] > th["hr_high"])
            | (tl["resp_rate"] > th["rr_high"]) | (tl["spo2"] < th["spo2_low"])
            | (tl["lactate"] > th["lactate_high"])).astype(int)


def build_labels(timeline: pd.DataFrame, cohort: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    lc = cfg["labels"]
    H, K = lc["horizon_hours"], lc["sustained_hours"]
    tl = timeline.sort_values(["stay_id", "hour"]).reset_index(drop=True)
    stay = tl["stay_id"]
    out = tl[["stay_id", "hour"]].copy()
    out["unstable"] = unstable_flags(tl, lc["thresholds"])

    # sustained-instability onset: run of K unstable hours completes at h
    run = out.groupby("stay_id")["unstable"].rolling(K, min_periods=K).sum().reset_index(level=0, drop=True)
    sustained = (run == K).astype(int)
    prev_sustained = sustained.groupby(stay).shift(1).fillna(0)
    out["ev_instability"] = ((sustained == 1) & (prev_sustained == 0)).astype(int)

    meta = cohort.set_index("stay_id")
    vaso_h = stay.map(meta["vaso_start_hour"]) if "vaso_start_hour" in meta else pd.Series(np.nan, index=tl.index)
    death_h = stay.map(meta["death_hour"])
    out["ev_vasopressor"] = (out["hour"] == vaso_h).astype(int)
    out["ev_death"] = (out["hour"] == death_h).astype(int)
    out["event"] = out[["ev_instability", "ev_vasopressor", "ev_death"]].max(axis=1)

    # first event per stay (deaths past the last timeline hour still count)
    first_ev = out[out["event"] == 1].groupby("stay_id")["hour"].min()
    first_ev = pd.concat([first_ev, meta["death_hour"].dropna()], axis=1).min(axis=1)
    out["first_event_hour"] = stay.map(first_ev)

    # label: event onset in (t, t+H]; also record hours until it
    ev_hours = out["hour"].where(out["event"] == 1)
    nxt = ev_hours.groupby(stay).bfill().groupby(stay).shift(-1)   # next onset strictly after t
    # a death beyond the timeline grid
    dh = death_h.where(death_h > stay.map(out.groupby("stay_id")["hour"].max()))
    nxt = pd.concat([nxt, dh], axis=1).min(axis=1)
    out["hours_to_event"] = nxt - out["hour"]
    out["label"] = ((out["hours_to_event"] >= 1) & (out["hours_to_event"] <= H)).astype(int)

    # future observation coverage for negatives
    hr_obs = tl["heart_rate"].notna().astype(int)
    fut = sum(hr_obs.groupby(stay).shift(-k).fillna(0) for k in range(1, H + 1))
    on_vaso = tl["vasopressor"] if "vasopressor" in tl else 0

    eligible = (out["hour"] >= lc["min_history_hours"] - 1) & (out["unstable"] == 0) & (on_vaso == 0)
    if lc.get("censor_after_first_event", True):
        eligible &= ~(out["hour"] >= out["first_event_hour"])
    eligible &= (out["label"] == 1) | (fut >= lc["min_observed_hours_for_negative"])
    out["eligible"] = eligible.astype(int)
    return out


def build_forecast_targets(timeline: pd.DataFrame, targets: list[str], horizons: list[int]) -> pd.DataFrame:
    """target_<var>_h<k> = the observed hourly value of <var> k hours after t (NaN if not charted)."""
    tl = timeline.sort_values(["stay_id", "hour"]).reset_index(drop=True)
    out = tl[["stay_id", "hour"]].copy()
    g = tl.groupby("stay_id")
    for v in targets:
        for k in horizons:
            out[f"target_{v}_h{k}"] = g[v].shift(-k)
    return out
