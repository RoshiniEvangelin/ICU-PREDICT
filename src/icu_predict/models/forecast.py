"""Forecast every vital sign for each of the next 6 hours, with an 80% range.

For each vital (heart rate, SBP, MAP, respiratory rate, SpO2, temperature) and
each hour ahead (+1h ... +6h), three LightGBM models are trained: the median
(the point forecast) and the 10th / 90th percentiles (the range). Together they
draw a "fan": narrow near now, wider further out.

Every forecast is compared on the test set with the "persistence" baseline —
assuming the value k hours from now equals the last observed value — which is
hard to beat in ICU data, especially 1-2 hours out.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from icu_predict.config import forecast_horizons, load_config
from icu_predict.models.train import get_feature_list

QUANTILES = {"q10": 0.1, "q50": 0.5, "q90": 0.9}


def _band(ms: dict, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(low, median, high) for a fitted quantile trio, including its conformal margin."""
    lo, mid, hi = np.sort(np.vstack([ms[q].predict(X) for q in ("q10", "q50", "q90")]), axis=0)
    m = ms.get("margin", 0.0)
    return lo - m, mid, np.maximum(hi + m, mid)


def run(cfg: dict) -> pd.DataFrame:
    from lightgbm import LGBMRegressor
    proc, out = Path(cfg["paths"]["processed_dir"]), Path(cfg["paths"]["outputs_dir"])
    ds = pd.read_parquet(proc / "dataset.parquet")
    splits = pd.read_csv(out / "reports/splits.csv")[["stay_id", "split"]]
    ds = ds.merge(splits, on="stay_id")
    ds = ds[ds["hour"] >= cfg["labels"]["min_history_hours"] - 1]
    feats = get_feature_list(ds.drop(columns="split"))
    horizons = forecast_horizons(cfg)
    n_est = cfg["forecast"].get("n_estimators", 300)

    models, rows = {}, []
    for v in cfg["forecast"]["targets"]:
        for k in horizons:
            t = f"target_{v}_h{k}"
            d = ds[ds[t].notna()]
            tr, va, te = (d[d["split"] == s] for s in ("train", "val", "test"))
            if len(tr) < 200 or len(te) < 20:
                print(f"[forecast] skipping {v} +{k}h: not enough data")
                continue
            models.setdefault(v, {})[k] = {}
            for qn, q in QUANTILES.items():  # noqa: B007
                m = LGBMRegressor(objective="quantile", alpha=q, n_estimators=n_est, learning_rate=0.05,
                                  num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1,
                                  colsample_bytree=0.7, verbose=-1, n_jobs=-1)
                m.fit(tr[feats], tr[t])
                models[v][k][qn] = m
            # Conformal widening (CQR): stretch the range by the amount needed so that 80% of
            # validation patients' real values fall inside it.
            margin = 0.0
            if len(va) >= 20:
                vlo, _, vhi = _band(models[v][k], va[feats])
                score = np.maximum(vlo - va[t].to_numpy(), va[t].to_numpy() - vhi)
                n = len(score)
                margin = float(np.quantile(score, min(1.0, np.ceil((n + 1) * 0.8) / n)))
            models[v][k]["margin"] = margin
            y = te[t].to_numpy()
            lo, mid, hi = _band(models[v][k], te[feats])
            persist = te[f"{v}_last"].fillna(tr[t].median()).to_numpy()
            mae_m, mae_p = float(np.mean(np.abs(y - mid))), float(np.mean(np.abs(y - persist)))
            rows.append({"target": v, "hours_ahead": k, "n_test": len(te), "mae_model": mae_m,
                         "mae_persistence": mae_p, "improvement_vs_persistence": 1 - mae_m / mae_p,
                         "interval_coverage_80": float(np.mean((y >= lo) & (y <= hi)))})
        print(f"[forecast] trained {v} for +{horizons[0]}..+{horizons[-1]}h")
    res = pd.DataFrame(rows)
    (out / "models").mkdir(parents=True, exist_ok=True)
    joblib.dump({"models": models, "features": feats, "horizons": horizons,
                 "horizon_hours": cfg["labels"]["horizon_hours"]}, out / "models/forecast_models.joblib")
    res.to_csv(out / "reports/forecast_metrics.csv", index=False)
    wide = res.pivot(index="target", columns="hours_ahead", values="mae_model").round(2)
    wide.columns = [f"+{c}h" for c in wide.columns]
    gain = res.pivot(index="target", columns="hours_ahead", values="improvement_vs_persistence")
    wide["vs persistence (avg)"] = (gain.mean(axis=1) * 100).round(1).astype(str) + "%"
    wide["80% range coverage"] = res.groupby("target")["interval_coverage_80"].mean().round(2)
    print("[forecast] TEST SET — mean absolute error by hours ahead\n" + wide.to_string())
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    run(load_config(ap.parse_args().config))
