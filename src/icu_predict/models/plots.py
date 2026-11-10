"""Evaluation figures for the test set (saved to outputs/figures/)."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve

from icu_predict.config import load_config
from icu_predict.models.common import contributions, pretty_name

COLORS = {"logistic_regression": "#4C78A8", "xgboost": "#E45756", "lightgbm": "#54A24B", "gru": "#8E6BBF", "news2_baseline": "#9D9D9D"}


def run(cfg: dict) -> None:
    out = Path(cfg["paths"]["outputs_dir"])
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    preds = pd.read_parquet(out / "reports/test_predictions.parquet")
    bundle = joblib.load(out / "models/deterioration_model.joblib")
    y = preds["label"].to_numpy()
    prob_cols = [c for c in preds.columns if c.startswith("prob_")]

    ds = pd.read_parquet(Path(cfg["paths"]["processed_dir"]) / "dataset.parquet")
    test = preds[["stay_id", "hour"]].merge(ds, on=["stay_id", "hour"])
    news = test["news2"].to_numpy() / 20

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for c in prob_cols + ["news2"]:
        name = c.replace("prob_", "") if c != "news2" else "news2_baseline"
        p = preds[c].to_numpy() if c != "news2" else news
        fpr, tpr, _ = roc_curve(y, p)
        pr, rc, _ = precision_recall_curve(y, p)
        style = "--" if name == "news2_baseline" else "-"
        axes[0].plot(fpr, tpr, style, color=COLORS.get(name), label=f"{name} (AUROC {roc_auc_score(y, p):.3f})")
        axes[1].plot(rc, pr, style, color=COLORS.get(name), label=f"{name} (AUPRC {average_precision_score(y, p):.3f})")
    axes[0].plot([0, 1], [0, 1], ":", color="gray")
    axes[0].set(xlabel="False positive rate", ylabel="Sensitivity", title="ROC — test set")
    axes[1].axhline(y.mean(), ls=":", color="gray", label=f"prevalence {y.mean():.3f}")
    axes[1].set(xlabel="Recall (sensitivity)", ylabel="Precision (PPV)", title="Precision-recall — test set")
    for a in axes:
        a.legend(fontsize=8)
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "roc_pr_curves.png", dpi=130)
    plt.close(fig)

    p = preds["prob"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    frac, mean = calibration_curve(y, p, n_bins=10, strategy="quantile")
    axes[0].plot(mean, frac, "o-", color=COLORS.get(bundle["model_name"]))
    axes[0].plot([0, p.max()], [0, p.max()], ":", color="gray")
    axes[0].set(xlabel="Predicted risk", ylabel="Observed rate", title=f"Calibration — {bundle['model_name']}")
    cm = confusion_matrix(y, (p >= bundle["threshold"]).astype(int))
    axes[1].imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        axes[1].text(j, i, f"{v:,}", ha="center", va="center", color="black", fontsize=12)
    axes[1].set(xticks=[0, 1], yticks=[0, 1], xticklabels=["No alert", "Alert"],
                yticklabels=["No event", "Event in 6h"], title=f"Confusion matrix (threshold {bundle['threshold']:.3f})")
    fig.tight_layout()
    fig.savefig(fig_dir / "calibration_confusion.png", dpi=130)
    plt.close(fig)

    X = test[bundle["features"]]
    contrib = contributions(bundle["model"].model, X.sample(min(3000, len(X)), random_state=0))
    imp = contrib.abs().mean().sort_values(ascending=True).tail(20)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh([pretty_name(f) for f in imp.index], imp.values, color="#4C78A8")
    ax.set(xlabel="Mean |contribution to log-odds|", title="Top 20 drivers of deterioration risk")
    fig.tight_layout()
    fig.savefig(fig_dir / "feature_importance.png", dpi=130)
    plt.close(fig)
    imp.sort_values(ascending=False).rename("mean_abs_contribution").to_csv(out / "reports/feature_importance.csv")
    example_patient_plot(cfg, preds, bundle)
    print(f"[plots] figures written to {fig_dir}")


def example_patient_plot(cfg: dict, preds: pd.DataFrame, bundle: dict) -> None:
    """One held-out patient a few hours before deterioration: history, hour-by-hour forecast fan, reality."""
    from icu_predict.inference.predictor import VITAL_LABELS, VITAL_UNITS, Predictor
    proc, out = Path(cfg["paths"]["processed_dir"]), Path(cfg["paths"]["outputs_dir"])
    pos = preds[(preds["label"] == 1) & (preds["hours_to_event"] >= 3)].sort_values("prob", ascending=False)
    if pos.empty:
        return
    sid, hour = int(pos.iloc[0]["stay_id"]), int(pos.iloc[0]["hour"])
    cohort, tl = pd.read_parquet(proc / "cohort.parquet"), pd.read_parquet(proc / "timeline.parquet")
    hist = tl[tl["stay_id"] == sid].sort_values("hour")
    r = Predictor(cfg).predict(hist[hist["hour"] <= hour].drop(columns="stay_id"),
                               cohort[cohort["stay_id"] == sid].iloc[0].to_dict())
    onset = hour + int(pos.iloc[0]["hours_to_event"])
    th = cfg["labels"]["thresholds"]
    danger = {"heart_rate": th["hr_high"], "sbp": th["sbp_low"], "map": th["map_low"],
              "resp_rate": th["rr_high"], "spo2": th["spo2_low"]}
    vit = [v for v in ["heart_rate", "sbp", "map", "resp_rate", "spo2", "temp_c"] if v in r["forecast"]]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    past = hist[(hist["hour"] <= hour) & (hist["hour"] >= hour - 12)]
    fut = hist[(hist["hour"] > hour) & (hist["hour"] <= hour + r["horizon_hours"])]
    for ax, v in zip(axes.flat, vit):
        f = r["forecast"][v]
        xs = [hour] + [hour + p["hours_ahead"] for p in f["path"]]
        start = f["current"] if f["current"] is not None else f["path"][0]["predicted"]
        ax.fill_between(xs, [start] + [p["low"] for p in f["path"]], [start] + [p["high"] for p in f["path"]],
                        color="#ef6c00", alpha=0.18, label="80% range")
        ax.plot(xs, [start] + [p["predicted"] for p in f["path"]], "-o", color="#ef6c00", ms=3, label="forecast")
        ax.plot(past["hour"], past[v], "-o", color="#1565c0", ms=3, label="so far")
        ax.plot(fut["hour"], fut[v], "o", color="#424242", ms=4, label="what really happened")
        if v in danger:
            ax.axhline(danger[v], color="#c62828", ls=":", lw=1, label="danger line")
        ax.axvline(hour, color="gray", lw=0.8)
        ax.axvline(onset, color="#c62828", ls="--", lw=1, label="deterioration onset")
        ax.set_title(f"{VITAL_LABELS[v]} ({VITAL_UNITS[v]})")
        ax.set_xlabel("ICU hour")
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=7, loc="best")
    fig.suptitle(f"Held-out patient at ICU hour {hour}: deterioration risk {r['risk_probability']:.0%} "
                 f"({r['risk_level']}) — deterioration actually began at hour {onset}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "figures/example_patient_next6h.png", dpi=130)
    plt.close(fig)
    (out / "reports/example_patient_report.md").write_text(r["detailed_report"], encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    run(load_config(ap.parse_args().config))
