"""Train, calibrate, select and evaluate 6-hour deterioration classifiers.

* Split by patient (train / val / test) — no patient in two splits.
* Fit Logistic Regression, XGBoost and LightGBM on train (early stopping on val).
* Pick the best by validation AUPRC, calibrate it with isotonic regression on val,
  and choose the alert threshold on val to hit the target sensitivity.
* Report everything on the untouched test set, alongside a NEWS2 baseline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from icu_predict.config import load_config
from icu_predict.features.build import feature_columns
from icu_predict.models.common import (CalibratedModel, bootstrap_ci, classification_metrics,
                                       event_level_metrics, fit_calibrator, patient_split,
                                       threshold_for_alert_budget, threshold_for_sensitivity)

NON_FEATURES = {"label", "eligible", "hours_to_event", "unstable", "subject_id", "split"}


def get_feature_list(ds: pd.DataFrame) -> list[str]:
    return [c for c in feature_columns(ds) if c not in NON_FEATURES and not c.startswith("target_")]


def make_model(name: str, pos_weight: float):
    if name == "logistic_regression":
        return Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                         ("scale", StandardScaler()),
                         ("clf", LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced"))])
    if name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=1500, learning_rate=0.03, max_depth=5, subsample=0.8,
                             colsample_bytree=0.7, min_child_weight=5, reg_lambda=1.0,
                             scale_pos_weight=pos_weight, eval_metric="aucpr", early_stopping_rounds=100,
                             n_jobs=-1, tree_method="hist")
    if name == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=1500, learning_rate=0.02, num_leaves=15, subsample=0.8,
                              subsample_freq=1, colsample_bytree=0.6, min_child_samples=100,
                              reg_lambda=5.0, scale_pos_weight=np.sqrt(pos_weight), n_jobs=-1, verbose=-1)
    raise ValueError(name)


def fit_model(name, model, Xtr, ytr, Xva, yva):
    if name == "xgboost":
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    elif name == "lightgbm":
        import lightgbm as lgb
        cb = [lgb.early_stopping(100, verbose=False)]
        try:   # lightgbm >= 4.6 API
            model.fit(Xtr, ytr, eval_X=(Xva,), eval_y=(yva,), eval_metric="average_precision", callbacks=cb)
        except TypeError:
            model.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="average_precision", callbacks=cb)
    else:
        model.fit(Xtr, ytr)
    return model


def _gru_cfg(cfg):
    g = {"enabled": True, "window": 12, "hidden": 64, "epochs": 30, "cv_epochs": 12}
    g.update(cfg["training"].get("gru", {}) or {})
    return g


def _gru_enabled(cfg) -> bool:
    from icu_predict.models import sequence
    if not _gru_cfg(cfg)["enabled"]:
        return False
    if not sequence.available():
        print("[train] GRU benchmark skipped: PyTorch is not installed (pip install torch).")
        return False
    return True


def cross_validate(cfg: dict, el: pd.DataFrame, feats: list[str], out: Path, ds: pd.DataFrame | None = None) -> pd.DataFrame:
    """Patient-grouped, label-stratified k-fold CV, repeated with different shuffles.

    With small cohorts (e.g. the 100-patient MIMIC-IV demo) a single held-out test
    set has a handful of patients and its metrics are mostly noise. CV scores every
    stay exactly once per repeat, which gives a far more stable estimate.
    """
    from sklearn.model_selection import StratifiedGroupKFold
    k, repeats = cfg["training"].get("cv_folds", 5), cfg["training"].get("cv_repeats", 3)
    y, groups = el["label"].to_numpy(), el["subject_id"].to_numpy()
    rows = []
    gru_on = ds is not None and _gru_enabled(cfg)
    if gru_on:
        from icu_predict.models.sequence import GRUClassifier, SequenceBuilder
        g = _gru_cfg(cfg)
        builder = SequenceBuilder(ds, g["window"])
    for rep in range(repeats):
        oof = {n: np.full(len(el), np.nan) for n in cfg["training"]["models"] + (["gru"] if gru_on else [])}
        for tr_idx, te_idx in StratifiedGroupKFold(k, shuffle=True, random_state=rep).split(el, y, groups):
            tr = el.iloc[tr_idx]
            pw = float((tr["label"] == 0).sum() / max((tr["label"] == 1).sum(), 1))
            for name in oof:
                if name == "gru":
                    builder.fit_scaler(tr["stay_id"].unique())
                    m = GRUClassifier(builder, hidden=g["hidden"], epochs=g["cv_epochs"], seed=rep)
                    m.fit(tr, tr["label"].to_numpy())
                    oof[name][te_idx] = m.predict_proba(el.iloc[te_idx])[:, 1]
                    continue
                m = make_model(name, pw)
                if name == "xgboost":
                    m.set_params(n_estimators=300, early_stopping_rounds=None)
                elif name == "lightgbm":
                    m.set_params(n_estimators=300)
                m.fit(tr[feats], tr["label"])
                oof[name][te_idx] = m.predict_proba(el.iloc[te_idx][feats])[:, 1]
        oof["news2_baseline"] = el["news2"].to_numpy() / 20.0
        for name, p in oof.items():
            rows.append({"repeat": rep, "model": name, "auroc": roc_auc_score(y, p),
                         "auprc": average_precision_score(y, p)})
    res = pd.DataFrame(rows)
    summary = res.groupby("model")[["auroc", "auprc"]].agg(["mean", "min", "max"]).round(3)
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index().sort_values("auprc_mean", ascending=False)
    summary.to_csv(out / "reports/cv_results.csv", index=False)
    print(f"[train] {k}-fold patient-grouped CV x {repeats} repeats on {el['stay_id'].nunique()} stays "
          f"(prevalence {y.mean():.2%}):\n" + summary.to_string(index=False))
    return summary


def run(cfg: dict) -> dict:
    proc, out = Path(cfg["paths"]["processed_dir"]), Path(cfg["paths"]["outputs_dir"])
    (out / "models").mkdir(parents=True, exist_ok=True)
    (out / "reports").mkdir(parents=True, exist_ok=True)
    ds = pd.read_parquet(proc / "dataset.parquet")
    sc = cfg["split"]
    ds["split"] = patient_split(ds["subject_id"], sc["test_size"], sc["val_size"], sc["seed"])
    ds[["stay_id", "subject_id", "split"]].drop_duplicates().to_csv(out / "reports/splits.csv", index=False)

    feats = get_feature_list(ds)
    el = ds[ds["eligible"] == 1]
    tr, va, te = (el[el["split"] == s] for s in ("train", "val", "test"))
    for s, d in (("train", tr), ("val", va), ("test", te)):
        print(f"[train] {s:5s}: {len(d):7,} rows, {d['stay_id'].nunique():5,} stays, positive {d['label'].mean():.2%}")
    pos_weight = float((tr["label"] == 0).sum() / max((tr["label"] == 1).sum(), 1))

    n_stays = el["stay_id"].nunique()
    run_cv = cfg["training"].get("run_cv", "auto")
    cv_summary = None
    if run_cv is True or (run_cv == "auto" and n_stays < cfg["training"].get("cv_auto_below_stays", 500)):
        cv_summary = cross_validate(cfg, el, feats, out, ds)
        if te["stay_id"].nunique() < 50:
            print(f"[train] NOTE: the test set has only {te['stay_id'].nunique()} stays — trust the CV numbers "
                  f"above, not the single test-set table below.")

    fitted, rows = {}, []
    for name in cfg["training"]["models"]:
        m = fit_model(name, make_model(name, pos_weight), tr[feats], tr["label"], va[feats], va["label"])
        pv = m.predict_proba(va[feats])[:, 1]
        fitted[name] = m
        rows.append({"model": name, "val_auroc": roc_auc_score(va["label"], pv),
                     "val_auprc": average_precision_score(va["label"], pv)})
        print(f"[train] {name:20s} val AUROC {rows[-1]['val_auroc']:.3f}  AUPRC {rows[-1]['val_auprc']:.3f}")

    metric = cfg["training"]["selection_metric"]
    if cv_summary is not None:   # small cohort: CV over every stay is a far steadier guide than a tiny val set
        cands = cv_summary[cv_summary["model"].isin(fitted)]
        best = cands.sort_values(f"{metric}_mean", ascending=False).iloc[0]["model"]
        print(f"[train] selected by cross-validated {metric}: {best}")
    else:
        best = max(rows, key=lambda r: r["val_" + metric])["model"]
        print(f"[train] selected by validation {metric}: {best}")

    # Calibrate every model on val so test comparisons are like-for-like; pick threshold on val.
    tc = cfg["training"]
    target_sens = tc["alert_target_sensitivity"]
    strategy, calib = tc.get("alert_strategy", "sensitivity"), tc.get("calibration", "isotonic")

    def pick_threshold(y, p):
        if strategy == "budget":
            return threshold_for_alert_budget(p, tc.get("alerts_per_patient_day", 1.5))
        return threshold_for_sensitivity(y, p, target_sens)

    results, test_preds = [], te[["stay_id", "subject_id", "hour", "label", "hours_to_event"]].copy()
    bundles = {}
    for name, m in fitted.items():
        cal = CalibratedModel(m, fit_calibrator(m.predict_proba(va[feats])[:, 1], va["label"].to_numpy(), calib), name)
        thr = pick_threshold(va["label"].to_numpy(), cal.predict_proba(va[feats]))
        p = cal.predict_proba(te[feats])
        res = {"model": name, **classification_metrics(te["label"].to_numpy(), p, thr)}
        res.update(event_level_metrics(test_preds.assign(prob=p), thr, cfg["labels"]["horizon_hours"]))
        results.append(res)
        test_preds[f"prob_{name}"] = p
        watch = (threshold_for_alert_budget(cal.predict_proba(va[feats]), tc.get("watch_per_patient_day", 4.0))
                 if strategy == "budget" else thr / 2)
        bundles[name] = (cal, thr, min(watch, thr))

    # GRU sequence model — benchmark only (same rows, splits, calibration and threshold rule)
    if _gru_enabled(cfg):
        from icu_predict.models.sequence import GRUClassifier, SequenceBuilder
        g = _gru_cfg(cfg)
        builder = SequenceBuilder(ds, g["window"]).fit_scaler(tr["stay_id"].unique())
        gru = GRUClassifier(builder, hidden=g["hidden"], epochs=g["epochs"])
        gru.fit(tr, tr["label"].to_numpy(), va, va["label"].to_numpy())
        pv = gru.predict_proba(va)[:, 1]
        rows.append({"model": "gru", "val_auroc": roc_auc_score(va["label"], pv), "val_auprc": average_precision_score(va["label"], pv)})
        cal_g = fit_calibrator(pv, va["label"].to_numpy(), calib)
        thr_g = pick_threshold(va["label"].to_numpy(), cal_g.predict(pv))
        p = np.clip(cal_g.predict(gru.predict_proba(te)[:, 1]), 1e-4, 1 - 1e-4)
        res = {"model": "gru", **classification_metrics(te["label"].to_numpy(), p, thr_g)}
        res.update(event_level_metrics(test_preds.assign(prob=p), thr_g, cfg["labels"]["horizon_hours"]))
        results.append(res)
        test_preds["prob_gru"] = p
        print(f"[train] {'gru (benchmark)':20s} val AUROC {rows[-1]['val_auroc']:.3f}  AUPRC {rows[-1]['val_auprc']:.3f}")

    # NEWS2 baseline: the bedside early-warning score, no ML. Clinical trigger at NEWS2 >= 5.
    news = te["news2"].to_numpy() / 20.0
    res = {"model": "news2_baseline", **classification_metrics(te["label"].to_numpy(), news, 5 / 20.0)}
    res.update(event_level_metrics(test_preds.assign(prob=news), 5 / 20.0, cfg["labels"]["horizon_hours"]))
    results.append(res)

    comp = pd.DataFrame(results).merge(pd.DataFrame(rows), on="model", how="left")
    comp.to_csv(out / "reports/model_comparison.csv", index=False)

    cal, thr, watch_thr = bundles[best]
    p_best = test_preds[f"prob_{best}"].to_numpy()
    ci = bootstrap_ci(te["label"].to_numpy(), p_best, te["stay_id"].to_numpy(), cfg["training"]["bootstrap_iterations"])
    best_row = comp[comp["model"] == best].iloc[0].to_dict()
    summary = {"selected_model": best, "n_features": len(feats),
               "horizon_hours": cfg["labels"]["horizon_hours"],
               "test": {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) and v is not None else v)
                        for k, v in best_row.items()}, **ci,
               "data": {"train_rows": len(tr), "val_rows": len(va), "test_rows": len(te),
                        "test_stays": int(te["stay_id"].nunique())}}
    with open(out / "reports/metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    test_preds["prob"] = p_best
    test_preds.to_parquet(out / "reports/test_predictions.parquet", index=False)

    joblib.dump({"model": cal, "threshold": thr, "watch_threshold": watch_thr, "features": feats, "model_name": best,
                 "horizon_hours": cfg["labels"]["horizon_hours"], "config": cfg,
                 "train_positive_rate": float(tr["label"].mean())},
                out / "models/deterioration_model.joblib")
    show = comp.set_index("model")[["auroc", "auprc", "brier", "sensitivity", "specificity", "ppv",
                                     "episode_sensitivity", "median_lead_time_hours", "alerts_per_patient_day"]]
    print("[train] TEST SET RESULTS\n" + show.round(3).to_string())
    print(f"[train] {best}: AUROC 95% CI {np.round(ci['auroc_ci'], 3)}, AUPRC 95% CI {np.round(ci['auprc_ci'], 3)}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    run(load_config(ap.parse_args().config))
