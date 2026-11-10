"""FastAPI service for ICU-PREDICT.

Run:  python -m uvicorn app.api:app --host 127.0.0.1 --port 8000
Docs: http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from functools import lru_cache  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse, Response  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from icu_predict import __version__  # noqa: E402
from icu_predict.config import DYNAMIC_VARS, load_config  # noqa: E402
from icu_predict.inference.predictor import Predictor  # noqa: E402

app = FastAPI(title="ICU-PREDICT API", version=__version__,
              description="6-hour ICU deterioration risk and vital-sign forecasts. Research prototype — "
                          "not a medical device.")
CFG = load_config()
_predictor: Optional[Predictor] = None


def predictor() -> Predictor:
    global _predictor
    if _predictor is None:
        try:
            _predictor = Predictor(CFG)
        except FileNotFoundError as e:
            raise HTTPException(503, str(e))
    return _predictor


class HourlyObservation(BaseModel):
    heart_rate: Optional[float] = Field(None, description="bpm")
    sbp: Optional[float] = Field(None, description="systolic BP, mmHg")
    dbp: Optional[float] = None
    map: Optional[float] = Field(None, description="mean arterial pressure, mmHg")
    resp_rate: Optional[float] = None
    spo2: Optional[float] = None
    temp_c: Optional[float] = Field(None, description="°C")
    gcs_total: Optional[float] = None
    fio2: Optional[float] = None
    lactate: Optional[float] = None
    creatinine: Optional[float] = None
    wbc: Optional[float] = None
    hemoglobin: Optional[float] = None
    platelets: Optional[float] = None
    potassium: Optional[float] = None
    sodium: Optional[float] = None
    bicarbonate: Optional[float] = None
    glucose: Optional[float] = None
    bun: Optional[float] = None
    vasopressor: int = 0


class PatientInfo(BaseModel):
    age: float = 65
    male: int = 0
    emergency_admission: int = 1
    first_careunit: str = "Medical Intensive Care Unit (MICU)"


class PredictRequest(BaseModel):
    patient: PatientInfo = PatientInfo()
    history: list[HourlyObservation] = Field(..., min_length=1,
                                             description="Hourly observations since ICU admission, oldest first; "
                                                         "use null for anything not measured that hour.")

    model_config = {"json_schema_extra": {"example": {
        "patient": {"age": 71, "male": 1, "emergency_admission": 1,
                    "first_careunit": "Medical Intensive Care Unit (MICU)"},
        "history": [{"heart_rate": 86 + 3 * i, "sbp": 124 - 3 * i, "dbp": 70 - 1 * i, "resp_rate": 17 + i,
                     "spo2": 97 - 0.5 * i, "temp_c": 37.2 + 0.1 * i, "lactate": 1.8 if i == 0 else None}
                    for i in range(8)]}}}


STATIC = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/", include_in_schema=False)
def ui():
    """Central monitoring station (all beds at once) — the main UI."""
    return FileResponse(STATIC / "station.html")


@app.get("/station", include_in_schema=False)
def station():
    return FileResponse(STATIC / "station.html")


@app.get("/dashboard", include_in_schema=False)
def dashboard():
    """Optional light single-patient dashboard."""
    return FileResponse(STATIC / "dashboard.html")


@app.get("/api/info")
def root():
    return {"service": "ICU-PREDICT", "version": __version__, "docs": "/docs",
            "disclaimer": "Research prototype trained on MIMIC-IV-format data; not for clinical use."}


@app.get("/health")
def health():
    try:
        p = predictor()
        return {"status": "ok", "model": p.clf["model_name"], "threshold": p.threshold,
                "horizon_hours": p.horizon, "forecast_targets": list(p.fc["models"]) if p.fc else []}
    except HTTPException as e:
        return {"status": "model_not_trained", "detail": e.detail}


@app.get("/features")
def features():
    p = predictor()
    return {"input_variables": DYNAMIC_VARS + ["vasopressor"], "model_features": p.features}


@app.post("/predict")
def predict(req: PredictRequest):
    p = predictor()
    hist = pd.DataFrame([h.model_dump() for h in req.history])
    return p.predict(hist, req.patient.model_dump())


# --- demo helpers: score held-out test-set stays -------------------------------
def _demo_tables():
    proc, out = Path(CFG["paths"]["processed_dir"]), Path(CFG["paths"]["outputs_dir"])
    try:
        cohort = pd.read_parquet(proc / "cohort.parquet")
        timeline = pd.read_parquet(proc / "timeline.parquet")
        splits = pd.read_csv(out / "reports/splits.csv")
    except FileNotFoundError:
        raise HTTPException(503, "Processed data not found — run the pipeline first.")
    return cohort, timeline, splits


@app.get("/stays")
def list_stays(limit: int = 50):
    cohort, _, splits = _demo_tables()
    test = cohort[cohort["stay_id"].isin(splits.loc[splits["split"] == "test", "stay_id"])]
    cols = ["stay_id", "age", "male", "first_careunit", "n_hours"]
    return test[cols].head(limit).to_dict(orient="records")


@app.get("/stays/{stay_id}/predict")
def predict_stay(stay_id: int, hour: Optional[int] = None):
    cohort, timeline, _ = _demo_tables()
    row = cohort[cohort["stay_id"] == stay_id]
    if row.empty:
        raise HTTPException(404, f"stay {stay_id} not found")
    hist = timeline[timeline["stay_id"] == stay_id].sort_values("hour")
    if hour is not None:
        hist = hist[hist["hour"] <= hour]
    if hist.empty:
        raise HTTPException(400, "no data at or before that hour")
    static = row.iloc[0].to_dict()
    res = predictor().predict(hist.drop(columns=["stay_id"]), static)
    res["stay_id"], res["hour"] = stay_id, int(hist["hour"].iloc[-1])
    return res


# --- central-station UI endpoints ----------------------------------------------
WARD_VITALS = ["heart_rate", "sbp", "dbp", "map", "resp_rate", "spo2", "temp_c"]
LAB_VARS = ["lactate", "glucose", "creatinine", "wbc", "gcs_total"]
MAX_BEDS = 16


@lru_cache(maxsize=1)
def _ward_tables():
    cohort, timeline, splits = _demo_tables()
    labels_path = Path(CFG["paths"]["processed_dir"]) / "labels.parquet"
    labels = pd.read_parquet(labels_path) if labels_path.exists() else None
    test_ids = splits.loc[splits["split"] == "test", "stay_id"]
    beds = cohort[cohort["stay_id"].isin(test_ids)].sort_values("stay_id")
    if labels is not None and len(beds) > MAX_BEDS:   # big cohorts: show a mix incl. stays that deteriorated
        ev = labels.groupby("stay_id")["event"].max()
        beds = beds.assign(ev=beds["stay_id"].map(ev).fillna(0))
        beds = pd.concat([beds[beds["ev"] == 1].head(MAX_BEDS // 2), beds[beds["ev"] == 0]]).head(MAX_BEDS)
    beds = beds.head(MAX_BEDS).reset_index(drop=True)
    tl = timeline[timeline["stay_id"].isin(beds["stay_id"])].sort_values(["stay_id", "hour"]).copy()
    # what a monitor would show: last value, held for up to 4 h
    tl[WARD_VITALS] = tl.groupby("stay_id")[WARD_VITALS].ffill(limit=4)
    return beds, tl, labels


@lru_cache(maxsize=256)
def _trajectory(stay_id: int) -> pd.DataFrame:
    cohort, timeline, _ = _demo_tables()
    hist = timeline[timeline["stay_id"] == stay_id].sort_values("hour")
    static = cohort[cohort["stay_id"] == stay_id].iloc[0].to_dict()
    return predictor().predict_trajectory(hist.drop(columns="stay_id"), static).set_index("hour")["risk"]


def _bed_hour(i: int, max_hour: int, t: int) -> int:
    """Each bed joins the simulation at a different point in its stay, then advances with the clock."""
    start = 6 + (i * 7) % 18
    return int(min(start + t, max_hour))


def _clean(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), 1)


@app.get("/api/config")
def ui_config():
    p = predictor()
    src_file = Path(CFG["paths"]["outputs_dir"]) / "reports/data_source.json"
    try:
        import json
        source = json.loads(src_file.read_text(encoding="utf-8"))["source"]
    except Exception:
        source = "unknown"
    return {"thresholds": CFG["labels"]["thresholds"], "horizon_hours": p.horizon,
            "alert_threshold": p.threshold, "baseline_risk": p.clf["train_positive_rate"],
            "model": p.clf["model_name"], "data_source": source}


@app.on_event("startup")
def _warm_start():
    """With WARM_START=1 (hosted demo), score every bed in the background so the first visitor doesn't wait."""
    import os
    import threading
    if os.environ.get("WARM_START") != "1":
        return

    def warm():
        try:
            beds, _, _ = _ward_tables()
            for sid in beds["stay_id"]:
                _trajectory(int(sid))
            print("[startup] ward warmed up")
        except Exception as e:  # never block startup
            print(f"[startup] warm-up skipped: {e}")
    threading.Thread(target=warm, daemon=True).start()


@app.get("/api/ward")
def ward(t: int = 0):
    """Every monitored bed at simulation time t (hours since the simulation started)."""
    beds, tl, _ = _ward_tables()
    p = predictor()
    out = []
    for i, b in beds.iterrows():
        rows = tl[tl["stay_id"] == b["stay_id"]]
        max_h = int(rows["hour"].max())
        h = _bed_hour(i, max_h, t)
        traj = _trajectory(int(b["stay_id"]))
        risk = float(traj.loc[h])
        now = rows[rows["hour"] == h].iloc[0]
        out.append({
            "bed": f"{i + 1:02d}", "stay_id": int(b["stay_id"]), "age": int(b["age"]),
            "sex": "M" if b["male"] else "F", "unit": str(b["first_careunit"]),
            "hour": h, "max_hour": max_h, "end_of_record": h >= max_h,
            "risk": risk, "level": p.risk_level(risk), "alert": risk >= p.threshold,
            "on_vasopressors": bool(now["vasopressor"]),
            "vitals": {v: _clean(now[v]) for v in WARD_VITALS},
            "risk_trend": [float(x) for x in traj.loc[max(0, h - 11):h].to_list()],
        })
    return {"t": t, "beds": out}


@app.get("/api/patients")
def patients():
    """Patients available in the dashboard (held-out test stays, never seen in training)."""
    beds, tl, _ = _ward_tables()
    out = []
    for _, b in beds.iterrows():
        max_h = int(tl.loc[tl["stay_id"] == b["stay_id"], "hour"].max())
        out.append({"stay_id": int(b["stay_id"]), "age": int(b["age"]), "sex": "M" if b["male"] else "F",
                    "unit": str(b["first_careunit"]), "max_hour": max_h,
                    "default_hour": int(min(max_h, 24))})
    return out


@app.get("/api/patient/{stay_id}/report.pdf")
def patient_pdf(stay_id: int, hour: int, reveal: bool = False):
    """Download the full report (charts, tables, written summary) as a PDF."""
    from icu_predict.inference.report_pdf import build_pdf
    res = patient(stay_id, hour, reveal)
    pdf = build_pdf(res, CFG["labels"]["thresholds"])
    name = f"icu-report_stay-{stay_id}_hour-{res['hour_now']}.pdf"
    return Response(pdf, media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})


@app.get("/api/patient/{stay_id}")
def patient(stay_id: int, hour: int, reveal: bool = False):
    """Full prediction for one bed at a given ICU hour, plus the history needed to draw the charts."""
    cohort, timeline, _ = _demo_tables()
    row = cohort[cohort["stay_id"] == stay_id]
    if row.empty:
        raise HTTPException(404, f"stay {stay_id} not found")
    hist_all = timeline[timeline["stay_id"] == stay_id].sort_values("hour")
    hist = hist_all[hist_all["hour"] <= hour]
    if hist.empty:
        raise HTTPException(400, "no data at or before that hour")
    res = predictor().predict(hist.drop(columns=["stay_id"]), row.iloc[0].to_dict())
    past = hist_all[(hist_all["hour"] > hour - 24) & (hist_all["hour"] <= hour)]
    res["history"] = {"hour": past["hour"].astype(int).tolist(),
                      **{v: [_clean(x) for x in past[v]] for v in WARD_VITALS + LAB_VARS}}
    traj = _trajectory(stay_id)
    res["risk_history"] = {"hour": [int(x) for x in traj.loc[:hour].index], "risk": traj.loc[:hour].round(4).tolist()}
    if reveal:   # retrospective teaching mode: show what really happened next
        fut = hist_all[(hist_all["hour"] > hour) & (hist_all["hour"] <= hour + res["horizon_hours"])]
        res["actual"] = {"hour": fut["hour"].astype(int).tolist(), **{v: [_clean(x) for x in fut[v]] for v in WARD_VITALS}}
        _, _, labels = _ward_tables()
        if labels is not None:
            lab = labels[(labels["stay_id"] == stay_id) & (labels["event"] == 1) & (labels["hour"] > hour)]
            res["actual"]["next_event_hour"] = int(lab["hour"].min()) if len(lab) else None
            win = labels[(labels["stay_id"] == stay_id) & (labels["hour"] > hour)
                         & (labels["hour"] <= hour + res["horizon_hours"])]
            res["actual"]["unstable_hours"] = int(win["unstable"].sum())
            res["actual"]["vasopressor_hours"] = int(fut["vasopressor"].sum())
            res["actual"]["died"] = bool(win["ev_death"].sum())
    meta = row.iloc[0]
    res["patient"] = {"stay_id": stay_id, "age": int(meta["age"]), "sex": "M" if meta["male"] else "F",
                      "unit": str(meta["first_careunit"]), "admission_type": str(meta.get("admission_type", "")),
                      "max_hour": int(hist_all["hour"].max())}
    return res
