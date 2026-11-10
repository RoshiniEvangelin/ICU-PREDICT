"""Shared model utilities: patient-level splits, calibration, metrics, explanations."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, f1_score,
                             roc_auc_score)


def patient_split(subject_ids: pd.Series, test_size: float, val_size: float, seed: int) -> pd.Series:
    """Assign every row to train/val/test by PATIENT, so no patient appears in two splits."""
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(subject_ids.unique()))
    rng.shuffle(uniq)
    n_test, n_val = int(len(uniq) * test_size), int(len(uniq) * val_size)
    which = {s: "test" for s in uniq[:n_test]}
    which.update({s: "val" for s in uniq[n_test:n_test + n_val]})
    which.update({s: "train" for s in uniq[n_test + n_val:]})
    return subject_ids.map(which)


class CalibratedModel:
    """Wraps a fitted classifier with an isotonic calibrator fitted on validation data."""

    def __init__(self, model, calibrator: IsotonicRegression, name: str):
        self.model, self.calibrator, self.name = model, calibrator, name

    def raw_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.clip(self.calibrator.predict(self.raw_proba(X)), 1e-4, 1 - 1e-4)


class PlattCalibrator:
    """Sigmoid (Platt) calibration on the log-odds of the raw score.

    Smooth, so patients get distinct risk values. Isotonic regression is more flexible but on small
    validation sets it collapses scores into a few flat steps (many patients get *exactly* the same risk).
    """

    def __init__(self):
        from sklearn.linear_model import LogisticRegression
        self.lr = LogisticRegression(C=1e4)

    @staticmethod
    def _logit(p):
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def fit(self, raw, y):
        self.lr.fit(self._logit(raw), y)
        return self

    def predict(self, raw):
        return self.lr.predict_proba(self._logit(raw))[:, 1]


def fit_calibrator(raw_val: np.ndarray, y_val: np.ndarray, method: str = "auto"):
    """'isotonic', 'sigmoid', or 'auto' (isotonic only when validation has >= 300 positive hours)."""
    if method == "auto":
        method = "isotonic" if int(np.sum(y_val)) >= 300 else "sigmoid"
    if method == "sigmoid":
        return PlattCalibrator().fit(raw_val, y_val)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    iso.fit(raw_val, y_val)
    return iso


def threshold_for_alert_budget(p: np.ndarray, alerts_per_patient_day: float) -> float:
    """Lowest threshold whose alert rate stays within the budget (alerts per 24 patient-hours).

    Clinicians tolerate a limited number of alarms; picking the threshold this way fixes the alarm burden
    and lets sensitivity be whatever the model can achieve at that burden.
    """
    budget = alerts_per_patient_day / 24.0
    best = float(np.max(p)) + 1e-9
    for v in np.unique(p)[::-1]:
        if np.mean(p >= v) > budget:
            break
        best = float(v)
    return best


def threshold_for_sensitivity(y: np.ndarray, p: np.ndarray, target: float) -> float:
    """Highest threshold whose sensitivity on (y, p) is still >= target."""
    pos = np.sort(p[y == 1])[::-1]
    if len(pos) == 0:
        return 0.5
    k = int(np.ceil(target * len(pos))) - 1
    return float(pos[min(max(k, 0), len(pos) - 1)])


def expected_calibration_error(y, p, bins: int = 10) -> float:
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def classification_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "auroc": roc_auc_score(y, p),
        "auprc": average_precision_score(y, p),
        "prevalence": float(y.mean()),
        "brier": brier_score_loss(y, np.clip(p, 0, 1)),
        "ece": expected_calibration_error(y, p),
        "threshold": float(threshold),
        "sensitivity": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "ppv": tp / max(tp + fp, 1),
        "npv": tn / max(tn + fn, 1),
        "f1": f1_score(y, pred, zero_division=0),
        "alert_rate": float(pred.mean()),
    }


def bootstrap_ci(y, p, groups, n: int = 200, seed: int = 0) -> dict:
    """95% CIs for AUROC/AUPRC resampling whole stays (rows within a stay are correlated)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"y": y, "p": p, "g": groups})
    by_g = {g: d for g, d in df.groupby("g")}
    keys = np.array(list(by_g))
    aurocs, auprcs = [], []
    for _ in range(n):
        s = pd.concat([by_g[k] for k in rng.choice(keys, len(keys))])
        if s["y"].nunique() < 2:
            continue
        aurocs.append(roc_auc_score(s["y"], s["p"]))
        auprcs.append(average_precision_score(s["y"], s["p"]))
    q = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return {"auroc_ci": q(aurocs), "auprc_ci": q(auprcs)}


def event_level_metrics(rows: pd.DataFrame, threshold: float, horizon: int) -> dict:
    """Per deterioration episode: was there an alert in the `horizon` hours before onset, and how early?

    rows: eligible prediction rows with stay_id, hour, prob, label, hours_to_event.
    """
    pos = rows[rows["label"] == 1].copy()
    pos["onset"] = pos["hour"] + pos["hours_to_event"]
    detected, lead = 0, []
    episodes = pos.groupby(["stay_id", "onset"])
    for _, ep in episodes:
        alerts = ep[ep["prob"] >= threshold]
        if len(alerts):
            detected += 1
            lead.append(float(alerts["hours_to_event"].max()))
    n_ep = episodes.ngroups
    patient_days = len(rows) / 24
    return {
        "episodes": int(n_ep),
        "episodes_detected": int(detected),
        "episode_sensitivity": detected / max(n_ep, 1),
        "median_lead_time_hours": float(np.median(lead)) if lead else None,
        "alerts_per_patient_day": float((rows["prob"] >= threshold).sum() / max(patient_days, 1e-9)),
    }


# ---------------------------------------------------------------------------
# Explanations
# ---------------------------------------------------------------------------
def contributions(model, X: pd.DataFrame) -> pd.DataFrame:
    """Per-row, per-feature contributions to the model's log-odds (SHAP values for trees)."""
    from sklearn.pipeline import Pipeline
    if isinstance(model, Pipeline):  # logistic regression: coef * standardised value
        Z = model[:-1].transform(X)
        coef = model[-1].coef_[0]
        return pd.DataFrame(Z * coef, columns=X.columns, index=X.index)
    name = type(model).__name__
    if name.startswith("XGB"):
        import xgboost as xgb
        c = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
    elif name.startswith("LGBM"):
        c = model.predict(X, pred_contrib=True)
    else:
        raise TypeError(f"no contribution method for {name}")
    return pd.DataFrame(np.asarray(c)[:, :-1], columns=X.columns, index=X.index)


_VAR_NAMES = {
    "heart_rate": "Heart rate", "sbp": "Systolic BP", "dbp": "Diastolic BP", "map": "Mean arterial pressure",
    "resp_rate": "Respiratory rate", "spo2": "SpO2", "temp_c": "Temperature", "gcs_total": "GCS",
    "fio2": "FiO2", "lactate": "Lactate", "creatinine": "Creatinine", "wbc": "WBC", "hemoglobin": "Hemoglobin",
    "platelets": "Platelets", "potassium": "Potassium", "sodium": "Sodium", "bicarbonate": "Bicarbonate",
    "glucose": "Glucose", "bun": "BUN",
}
_SUFFIX = {
    "_last": "", "_mean6": " (6h mean)", "_min6": " (6h min)", "_max6": " (6h max)", "_std6": " (6h variability)",
    "_delta1": " change over 1h", "_delta6": " change over 6h", "_hrs_since": ": hours since measured",
    "_change": " change since last test",
}
_OTHER = {"shock_index": "Shock index (HR/SBP)", "pulse_pressure": "Pulse pressure", "news2": "NEWS2 score",
          "vasopressor_now": "On vasopressors", "vasopressor_hours": "Hours on vasopressors",
          "hours_in_icu": "Hours in ICU", "age": "Age", "male": "Male sex",
          "emergency_admission": "Emergency admission"}


def pretty_name(feature: str) -> str:
    if feature in _OTHER:
        return _OTHER[feature]
    if feature.startswith("unit_"):
        return f"Care unit: {feature[5:].upper()}"
    for suf, txt in _SUFFIX.items():
        if feature.endswith(suf):
            base = feature[: -len(suf)]
            return _VAR_NAMES.get(base, base) + txt
    return feature
