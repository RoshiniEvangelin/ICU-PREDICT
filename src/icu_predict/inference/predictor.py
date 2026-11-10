"""Load trained artifacts and score a patient's hourly history.

Used by both the FastAPI service and the Streamlit dashboard so they always
share one feature definition and one model.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from icu_predict.config import DYNAMIC_VARS, load_config
from icu_predict.features.build import build_features
from icu_predict.features.labels import unstable_flags
from icu_predict.models.common import contributions, pretty_name
from icu_predict.models.forecast import _band

VITAL_UNITS = {"heart_rate": "bpm", "sbp": "mmHg", "map": "mmHg", "resp_rate": "/min", "spo2": "%", "temp_c": "°C"}
VITAL_LABELS = {"heart_rate": "Heart rate", "sbp": "Systolic BP", "map": "MAP", "resp_rate": "Resp. rate",
                "spo2": "SpO2", "temp_c": "Temperature"}
# change smaller than this over 6 h is called "steady"
STEADY = {"heart_rate": 5, "sbp": 6, "map": 4, "resp_rate": 2, "spo2": 1, "temp_c": 0.3}
DECIMALS = {"temp_c": 1}


def danger_lines(th: dict) -> dict:
    """vital -> (direction, threshold, plain-English meaning). Same thresholds as the labels."""
    return {"map": ("below", th["map_low"], "low blood pressure — organs may not get enough blood flow"),
            "sbp": ("below", th["sbp_low"], "low systolic blood pressure (hypotension)"),
            "heart_rate": ("above", th["hr_high"], "very fast heart rate (tachycardia)"),
            "resp_rate": ("above", th["rr_high"], "very fast breathing — possible respiratory distress"),
            "spo2": ("below", th["spo2_low"], "low blood oxygen (hypoxemia)"),
            "temp_c": ("above", 38.3, "fever")}


def describe_factor(f: dict) -> str:
    """Turn a model feature + value into a short readable phrase."""
    feat, val = f["feature"], f["value"]
    if feat.endswith("_hrs_since"):
        base = pretty_name(feat[: -len("_hrs_since")] + "_last")
        if val is None or val >= 72:
            return f"{base} not measured during this stay"
        return f"{base} last measured {val:.0f} h ago"
    if val is None:
        return f"{f['name']}: not measured recently"
    if feat in ("male", "emergency_admission", "vasopressor_now") or feat.startswith("unit_"):
        return f"{f['name']}: {'yes' if val >= 0.5 else 'no'}"
    txt = f"{val:.2f}" if abs(val) < 10 else f"{val:.1f}" if abs(val) < 100 else f"{val:.0f}"
    if "." in txt:
        txt = txt.rstrip("0").rstrip(".")
    return f"{f['name']} = {txt}"


def fmt(v, val):
    return "–" if val is None or pd.isna(val) else f"{val:.{DECIMALS.get(v, 0)}f}"


class Predictor:
    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or load_config()
        mdir = Path(self.cfg["paths"]["outputs_dir"]) / "models"
        clf_path, fc_path = mdir / "deterioration_model.joblib", mdir / "forecast_models.joblib"
        if not clf_path.exists():
            raise FileNotFoundError(f"{clf_path} not found — run `python run_pipeline.py --demo` (or on MIMIC-IV) first.")
        self.clf = joblib.load(clf_path)
        self.fc = joblib.load(fc_path) if fc_path.exists() else None
        self.threshold = self.clf["threshold"]
        self.features = self.clf["features"]
        self.horizon = self.clf["horizon_hours"]

    # ------------------------------------------------------------------
    def featurize(self, history: pd.DataFrame, static: dict) -> pd.DataFrame:
        """history: one row per hour (oldest first) with any of DYNAMIC_VARS + 'vasopressor'."""
        tl = history.copy().reset_index(drop=True)
        tl["stay_id"] = static.get("stay_id", 0)
        if "hour" not in tl:
            tl["hour"] = np.arange(len(tl))
        for v in DYNAMIC_VARS:
            if v not in tl:
                tl[v] = np.nan
        tl["vasopressor"] = tl.get("vasopressor", 0)
        tl["vasopressor"] = tl["vasopressor"].fillna(0).astype(int)
        cohort = pd.DataFrame([{"stay_id": tl["stay_id"].iloc[0], "age": static.get("age", 65),
                                "male": int(static.get("male", 0)),
                                "emergency_admission": int(static.get("emergency_admission", 1)),
                                "first_careunit": static.get("first_careunit", "")}])
        return build_features(tl, cohort, self.cfg), tl

    def risk_level(self, p: float) -> str:
        if p >= self.threshold:
            return "HIGH"
        if p >= self.clf.get("watch_threshold", self.threshold / 2):
            return "MEDIUM"
        return "LOW"

    def predict(self, history: pd.DataFrame, static: dict, top_k: int = 6) -> dict:
        feats, tl = self.featurize(history, static)
        X = feats[self.features]
        x_last = X.iloc[[-1]]
        p = float(self.clf["model"].predict_proba(x_last)[0])
        contrib = contributions(self.clf["model"].model, x_last).iloc[0]
        top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(top_k)
        factors = [{"feature": f, "name": pretty_name(f), "value": None if pd.isna(x_last[f].iloc[0]) else float(x_last[f].iloc[0]),
                    "contribution": float(c), "direction": "raises risk" if c > 0 else "lowers risk"}
                   for f, c in top.items()]
        for f in factors:
            f["text"] = describe_factor(f)
        last = tl.iloc[[-1]]
        th = self.cfg["labels"]["thresholds"]
        currently_unstable = bool(unstable_flags(last.assign(map=feats["map_last"].iloc[[-1]].values), th).iloc[0])
        result = {
            "risk_probability": p,
            "risk_level": self.risk_level(p),
            "alert": p >= self.threshold,
            "alert_threshold": self.threshold,
            "baseline_risk": self.clf["train_positive_rate"],
            "relative_risk": p / max(self.clf["train_positive_rate"], 1e-6),
            "horizon_hours": self.horizon,
            "hours_of_history": int(len(tl)),
            "currently_unstable": currently_unstable,
            "on_vasopressors": bool(tl["vasopressor"].iloc[-1]),
            "top_factors": factors,
            "forecast": self.forecast(x_last, feats.iloc[-1]),
            "model_name": self.clf["model_name"],
        }
        # change over the last 6 h (or since admission if the stay is shorter)
        for v, f in result["forecast"].items():
            col = feats[f"{v}_last"]
            past = col.iloc[max(0, len(col) - 7):-1].dropna()
            if f["current"] is not None and len(past):
                f["change_last_6h"] = f["current"] - float(past.iloc[0])
                f["change_hours"] = int(len(col) - 1 - past.index[0] + col.index[0])
        result["hour_now"] = int(tl["hour"].iloc[-1])
        result["hourly_outlook"] = self.hourly_outlook(result["forecast"], result["hour_now"])
        result["summary"] = self.narrative(result)
        result["detailed_report"] = self.detailed_report(result, feats.iloc[-1])
        return result

    def forecast(self, x_last: pd.DataFrame, last_row: pd.Series) -> dict:
        """Per vital: current value and a path of (hours_ahead, predicted, low, high) for +1..+6 h.

        `predicted/low/high` at the top level are the furthest hour, for quick display.
        """
        if self.fc is None:
            return {}
        out = {}
        Xf = x_last.reindex(columns=self.fc["features"])
        for v, by_h in self.fc["models"].items():
            if not isinstance(next(iter(by_h.values())), dict):   # artifacts from older version
                by_h = {self.horizon: by_h}
            path = []
            for k in sorted(by_h):
                lo, mid, hi = (float(a[0]) for a in _band(by_h[k], Xf))
                if v == "spo2":
                    lo, mid, hi = min(lo, 100), min(mid, 100), min(hi, 100)
                path.append({"hours_ahead": int(k), "predicted": mid, "low": lo, "high": hi})
            cur = last_row.get(f"{v}_last")
            out[v] = {"label": VITAL_LABELS.get(v, v), "unit": VITAL_UNITS.get(v, ""),
                      "current": None if pd.isna(cur) else float(cur),
                      "change_last_6h": None if pd.isna(last_row.get(f"{v}_delta6")) else float(last_row.get(f"{v}_delta6")),
                      "path": path, **{k: path[-1][k] for k in ("predicted", "low", "high")}}
        return out

    def hourly_outlook(self, fc: dict, hour_now: int) -> list[dict]:
        """One entry per hour ahead: every vital's forecast plus any danger-line crossings."""
        lines = danger_lines(self.cfg["labels"]["thresholds"])
        hours = sorted({p["hours_ahead"] for f in fc.values() for p in f["path"]})
        outlook = []
        for k in hours:
            entry = {"hours_ahead": k, "icu_hour": hour_now + k, "vitals": {}, "warnings": []}
            for v, f in fc.items():
                p = next((p for p in f["path"] if p["hours_ahead"] == k), None)
                if p is None:
                    continue
                entry["vitals"][v] = p
                if v in lines:
                    d, thr, meaning = lines[v]
                    crossed = (lambda x: x < thr) if d == "below" else (lambda x: x > thr)
                    level = "likely" if crossed(p["predicted"]) else (
                        "possible" if crossed(p["low"] if d == "below" else p["high"]) else None)
                    if level:
                        entry["warnings"].append({"vital": v, "level": level, "direction": d, "threshold": thr,
                                                  "meaning": meaning})
            outlook.append(entry)
        return outlook

    def predict_trajectory(self, history: pd.DataFrame, static: dict) -> pd.DataFrame:
        """Risk at every hour of a stay (each hour uses only data up to that hour)."""
        feats, _ = self.featurize(history, static)
        p = self.clf["model"].predict_proba(feats[self.features])
        return pd.DataFrame({"hour": feats["hour"], "risk": p})

    # ------------------------------------------------------------------
    def narrative(self, r: dict) -> str:
        """Plain-language summary of the prediction (template-based, deterministic)."""
        pct = r["risk_probability"] * 100
        lines = []
        if r["currently_unstable"] or r["on_vasopressors"]:
            lines.append("The patient already meets instability criteria or is on vasopressors. The model is built "
                         "for patients who are currently stable, so treat this score with caution.")
        lines.append(f"Estimated risk of deterioration in the next {r['horizon_hours']} hours: {pct:.1f}% "
                     f"({r['risk_level']}; about {r['relative_risk']:.1f}x the average ICU patient-hour). "
                     + ("This crosses the alert threshold." if r["alert"] else "Below the alert threshold."))
        up = [f["name"] for f in r["top_factors"] if f["contribution"] > 0][:3]
        down = [f["name"] for f in r["top_factors"] if f["contribution"] < 0][:2]
        if up:
            lines.append("Main factors pushing risk up: " + ", ".join(up) + ".")
        if down:
            lines.append("Factors lowering risk: " + ", ".join(down) + ".")
        first = {}
        for h in r.get("hourly_outlook", []):
            for w in h["warnings"]:
                if w["level"] == "likely" and w["vital"] not in first:
                    first[w["vital"]] = (h["hours_ahead"], w)
        if first:
            lines.append("Expected to cross a danger line: " + "; ".join(
                f"{VITAL_LABELS[v]} {w['direction']} {w['threshold']:g} {VITAL_UNITS[v]} by +{k}h"
                for v, (k, w) in first.items()) + ".")
        elif r["forecast"]:
            lines.append(f"No vital sign is expected to cross a danger line in the next {r['horizon_hours']} hours.")
        lines.append("Research prototype — not for clinical decisions.")
        return " ".join(lines)

    # ------------------------------------------------------------------
    def detailed_report(self, r: dict, last_row: pd.Series) -> str:
        """Full plain-English Markdown report of what the next hours are expected to look like."""
        H, fc, now = r["horizon_hours"], r["forecast"], r["hour_now"]
        out = [f"## What to expect over the next {H} hours", f"*ICU hour {now} · forecasts for ICU hours "
               f"{now + 1}–{now + H}*", "", "### 1. Where the patient is right now"]
        for v, f in fc.items():
            if f["current"] is None:
                out.append(f"- **{f['label']}:** not measured recently")
                continue
            ch = f["change_last_6h"]
            trend = ""
            if ch is not None and not pd.isna(ch):
                trend = (f" — steady over the last {f.get('change_hours', 6)} h" if abs(ch) < STEADY[v] else
                         f" — {'up' if ch > 0 else 'down'} {fmt(v, abs(ch))} over the last {f.get('change_hours', 6)} h")
            out.append(f"- **{f['label']}:** {fmt(v, f['current'])} {f['unit']}{trend}")
        for lab, unit in (("lactate", "mmol/L"), ("creatinine", "mg/dL"), ("wbc", "K/µL")):
            val, since = last_row.get(f"{lab}_last"), last_row.get(f"{lab}_hrs_since")
            if val is not None and not pd.isna(val):
                out.append(f"- **{pretty_name(lab + '_last')}:** {val:.1f} {unit} (measured {int(since)} h ago)")
        if r["on_vasopressors"]:
            out.append("- **On vasopressors** (blood-pressure support medication)")

        pct, base = r["risk_probability"] * 100, r["baseline_risk"] * 100
        out += ["", f"### 2. Chance of getting worse in the next {H} hours",
                f"**{pct:.0f}%** — risk level **{r['risk_level']}**"
                + (" · **ALERT**" if r["alert"] else "") + ".",
                "",
                f"In plain terms: out of 100 patients who look like this right now, about **{pct:.0f}** would "
                f"deteriorate within {H} hours. For an average ICU patient-hour it's about {base:.0f} in 100, so "
                f"this patient is at roughly **{r['relative_risk']:.1f}×** the usual risk. Alerts fire at "
                f"{r['alert_threshold'] * 100:.1f}% or above.",
                "",
                "\"Deteriorate\" means at least one of: blood-pressure medication (vasopressor) is started, the "
                "patient dies, or vital signs stay in a danger zone for 2+ hours in a row (systolic BP < 90, "
                "MAP < 65, heart rate > 130, breathing > 30/min, SpO2 < 88% or lactate > 4)."]
        if r["currently_unstable"] or r["on_vasopressors"]:
            out += ["", "> ⚠️ The patient is already unstable or on vasopressors. The model was built to warn "
                        "*before* deterioration in stable patients, so this score is less reliable here."]

        if r["hourly_outlook"]:
            hrs = [h["hours_ahead"] for h in r["hourly_outlook"]]
            out += ["", "### 3. Hour by hour",
                    "Each cell is the most likely value, with the 80% range in brackets "
                    "(the real value lands inside about 8 times out of 10).", "",
                    "| | Now | " + " | ".join(f"+{k}h" for k in hrs) + " |",
                    "|---|---:|" + "---:|" * len(hrs)]
            for v, f in fc.items():
                cells = [f"{fmt(v, p['predicted'])} ({fmt(v, p['low'])}–{fmt(v, p['high'])})" for p in f["path"]]
                out.append(f"| {f['label']} ({f['unit']}) | {fmt(v, f['current'])} | " + " | ".join(cells) + " |")
            flags = []
            for h in r["hourly_outlook"]:
                w = [f"{'⚠️' if x['level'] == 'likely' else '·'} {VITAL_LABELS[x['vital']]}" for x in h["warnings"]]
                flags.append(", ".join(w) if w else "—")
            out.append("| **Danger-line flags** | | " + " | ".join(flags) + " |")
            out += ["", "⚠️ = the most likely value is past the danger line · `·` = possible: only the edge of the 80% range reaches it"]

            out += ["", "### 4. What to watch"]
            lines, watch = danger_lines(self.cfg["labels"]["thresholds"]), []
            for v, f in fc.items():
                if f["current"] is None:
                    continue
                end = f["path"][-1]["predicted"]
                delta = end - f["current"]
                word = "steady" if abs(delta) < STEADY[v] else ("rising" if delta > 0 else "falling")
                hits = [(h["hours_ahead"], w) for h in r["hourly_outlook"] for w in h["warnings"] if w["vital"] == v]
                likely = [k for k, w in hits if w["level"] == "likely"]
                if likely:
                    d, thr, meaning = lines[v]
                    watch.append(f"- **{f['label']}: likely {d} {thr:g} {f['unit']} from +{likely[0]}h** "
                                 f"({fmt(v, f['current'])} → {fmt(v, end)}). That would mean {meaning}.")
                elif hits:
                    d, thr, meaning = lines[v]
                    watch.append(f"- **{f['label']}:** expected {fmt(v, f['current'])} → {fmt(v, end)} {f['unit']} "
                                 f"({word}). Not expected to cross {thr:g}, but it's inside the range by "
                                 f"+{hits[0][0]}h — worth keeping an eye on.")
                elif word != "steady":
                    watch.append(f"- **{f['label']}:** {word}, {fmt(v, f['current'])} → {fmt(v, end)} {f['unit']} "
                                 f"by +{H}h, staying on the safe side of the danger line.")
            out += watch or [f"- Every vital sign is expected to stay roughly where it is and on the safe side "
                             f"of its danger line for the next {H} hours."]
            any_likely = any(w["level"] == "likely" for h in r["hourly_outlook"] for w in h["warnings"])
            possible = sorted({VITAL_LABELS[w["vital"]] for h in r["hourly_outlook"] for w in h["warnings"]})
            if r["alert"] and not any_likely:
                out += ["", "> **Why is the risk high if the vitals look OK?** The risk score and the vital "
                            "forecasts are separate models. The forecasts show the *most likely* path, which "
                            "tends to stay near normal; the risk score asks *how likely is any bad event*. "
                            + (f"Here the bad scenario is most likely through **{', '.join(possible)}** — the "
                               "low/high edge of their ranges reaches the danger line. " if possible else "")
                            + "Treat the alert as \"watch closely\", not \"it will happen\"."]
            elif not r["alert"] and any_likely:
                out += ["", "> **Note:** a vital is forecast to cross a danger line, but the overall risk is "
                            "below the alert threshold. A single brief crossing doesn't count as deterioration "
                            "(it must last 2+ hours), and the risk model weighs the whole picture."]

        out += ["", "### 5. Why the model gives this risk"]
        for f in r["top_factors"]:
            out.append(f"- {describe_factor(f)} → " + ("pushes risk **up** ↑" if f["contribution"] > 0
                                                     else "pulls risk **down** ↓"))
        if any(f["value"] is None or f["feature"].endswith("_hrs_since") for f in r["top_factors"]):
            out.append("- *Why does a missing test matter?* Doctors order tests when they're worried, so a test "
                       "that hasn't been ordered is itself a (weak) sign the team saw no reason to check.")
        out += ["", "### 6. How much to trust this",
                "- Ranges get wider further out because the future is less certain.",
                f"- Risk comes from a {r['model_name']} model trained on MIMIC-IV-format ICU data. It ranks risk "
                "better than chance but makes mistakes in both directions.",
                "- Research prototype — not for clinical decisions."]
        return "\n".join(out)
