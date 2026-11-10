"""Tests: label logic, no look-ahead leakage in features, and an end-to-end smoke run."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from icu_predict.config import DYNAMIC_VARS, load_config  # noqa: E402
from icu_predict.features.build import build_features  # noqa: E402
from icu_predict.features.labels import build_labels  # noqa: E402

CFG = load_config()


def make_stay(n=20, stay_id=1, **overrides):
    tl = pd.DataFrame({"stay_id": stay_id, "hour": np.arange(n)})
    base = dict(heart_rate=85.0, sbp=120.0, dbp=70.0, map=86.0, resp_rate=16.0, spo2=97.0, temp_c=37.0,
                gcs_total=15.0, fio2=np.nan)
    for v in DYNAMIC_VARS:
        tl[v] = base.get(v, np.nan)
    tl["vasopressor"] = 0
    for k, v in overrides.items():
        tl[k] = v
    return tl


def cohort_for(tl, **kw):
    row = dict(stay_id=tl["stay_id"].iloc[0], subject_id=1, age=60, male=1, emergency_admission=1,
               first_careunit="Medical Intensive Care Unit (MICU)", death_hour=np.nan, vaso_start_hour=np.nan)
    row.update(kw)
    return pd.DataFrame([row])


def test_sustained_hypotension_labels_preceding_six_hours():
    tl = make_stay(20)
    tl.loc[tl["hour"].isin([12, 13, 14]), ["sbp", "map"]] = [80.0, 55.0]
    lab = build_labels(tl, cohort_for(tl), CFG).set_index("hour")
    assert lab.loc[13, "event"] == 1                       # onset when the 2-hour run completes
    assert lab.loc[[7, 8, 9, 10, 11], "label"].eq(1).all()  # 13 - t in [1, 6] ...
    assert lab.loc[6, "label"] == 0                         # ... 7 hours out is not
    assert lab.loc[12, "eligible"] == 0                     # already unstable at 12
    assert lab.loc[15:, "eligible"].eq(0).all()             # censored after first event


def test_single_abnormal_hour_is_not_an_event():
    tl = make_stay(20)
    tl.loc[tl["hour"] == 10, "heart_rate"] = 150.0
    lab = build_labels(tl, cohort_for(tl), CFG)
    assert lab["event"].sum() == 0


def test_vasopressor_start_and_death_are_events():
    tl = make_stay(20)
    lab = build_labels(tl, cohort_for(tl, vaso_start_hour=10), CFG).set_index("hour")
    assert lab.loc[10, "event"] == 1 and lab.loc[4, "label"] == 1 and lab.loc[3, "label"] == 0
    lab = build_labels(tl, cohort_for(tl, death_hour=25), CFG).set_index("hour")   # death after last hour
    assert lab.loc[19, "label"] == 1 and lab.loc[18, "label"] == 0


def test_negatives_need_future_observations():
    tl = make_stay(20)
    tl.loc[tl["hour"] >= 10, ["heart_rate", "sbp", "dbp", "map", "resp_rate", "spo2"]] = np.nan
    lab = build_labels(tl, cohort_for(tl), CFG).set_index("hour")
    assert lab.loc[9, "eligible"] == 0          # nothing observed in the next 6 hours
    assert lab.loc[3, "eligible"] == 1


def test_features_do_not_look_ahead():
    rng = np.random.default_rng(0)
    tl = make_stay(30)
    for v in ["heart_rate", "sbp", "map", "resp_rate", "spo2", "lactate"]:
        tl[v] = rng.normal(tl[v].fillna(2.0), 5)
    co = cohort_for(tl)
    f1 = build_features(tl, co, CFG)
    tl2 = tl.copy()
    tl2.loc[tl2["hour"] > 15, ["heart_rate", "sbp", "map", "lactate"]] = [200.0, 60.0, 40.0, 12.0]
    f2 = build_features(tl2, co, CFG)
    pd.testing.assert_frame_equal(f1[f1["hour"] <= 15], f2[f2["hour"] <= 15])


@pytest.mark.slow
def test_end_to_end_demo(tmp_path):
    from icu_predict.data import extract
    from icu_predict.data.generate_demo import generate_demo
    from icu_predict.features import make_dataset
    from icu_predict.inference.predictor import Predictor
    from icu_predict.models import forecast, train
    cfg = load_config(overrides={"paths": {"demo_dir": str(tmp_path / "raw"), "processed_dir": str(tmp_path / "proc"),
                                           "outputs_dir": str(tmp_path / "out")},
                                 "training": {"bootstrap_iterations": 10}})
    generate_demo(cfg["paths"]["demo_dir"], n_patients=250, seed=1)
    extract.run(cfg, use_demo=True)
    ds = make_dataset.run(cfg)
    assert ds.loc[ds["eligible"] == 1, "label"].between(0, 1).all()
    summary = train.run(cfg)
    assert 0.5 < summary["test"]["auroc"] <= 1.0
    forecast.run(cfg)
    p = Predictor(cfg)
    hist = pd.DataFrame({"heart_rate": [90, 95, 100, 108, 115], "sbp": [120, 112, 104, 98, 93],
                         "dbp": [70, 66, 62, 58, 55], "resp_rate": [18, 20, 22, 24, 26], "spo2": [97, 96, 95, 94, 93]})
    r = p.predict(hist, {"age": 70, "male": 1})
    assert 0 <= r["risk_probability"] <= 1 and r["summary"] and "heart_rate" in r["forecast"]
    assert [h["hours_ahead"] for h in r["hourly_outlook"]] == [1, 2, 3, 4, 5, 6]
    path = r["forecast"]["heart_rate"]["path"]
    assert all(p["low"] <= p["predicted"] <= p["high"] for p in path)
    assert "Hour by hour" in r["detailed_report"]


def test_alert_budget_threshold_limits_alarm_rate():
    from icu_predict.models.common import threshold_for_alert_budget
    rng = np.random.default_rng(0)
    p = rng.beta(1, 15, 24_000)
    p[:2000] = 0.09                      # a big tie, like an isotonic plateau
    thr = threshold_for_alert_budget(p, alerts_per_patient_day=1.5)
    assert np.mean(p >= thr) <= 1.5 / 24 + 1e-9
    assert np.mean(p >= thr) > 1.0 / 24  # uses most of the budget, not far below it
