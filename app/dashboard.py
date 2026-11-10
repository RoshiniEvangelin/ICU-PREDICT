"""ICU-PREDICT dashboard.

Run:  python -m streamlit run app/dashboard.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

from icu_predict.config import load_config  # noqa: E402
from icu_predict.inference.predictor import VITAL_LABELS, VITAL_UNITS, Predictor, describe_factor, fmt  # noqa: E402

st.set_page_config(page_title="ICU-PREDICT", page_icon="🫀", layout="wide")
CFG = load_config()
PROC, OUT = Path(CFG["paths"]["processed_dir"]), Path(CFG["paths"]["outputs_dir"])
LEVEL_COLOR = {"LOW": "#2e7d32", "MEDIUM": "#ef6c00", "HIGH": "#c62828"}


@st.cache_resource
def get_predictor():
    return Predictor(CFG)


@st.cache_data
def load_data():
    cohort = pd.read_parquet(PROC / "cohort.parquet")
    timeline = pd.read_parquet(PROC / "timeline.parquet")
    labels = pd.read_parquet(PROC / "labels.parquet")
    splits = pd.read_csv(OUT / "reports/splits.csv")
    return cohort, timeline, labels, splits


@st.cache_data
def stay_trajectory(stay_id: int):
    cohort, timeline, _, _ = load_data()
    hist = timeline[timeline["stay_id"] == stay_id].sort_values("hour")
    static = cohort[cohort["stay_id"] == stay_id].iloc[0].to_dict()
    return get_predictor().predict_trajectory(hist.drop(columns="stay_id"), static)


def show_result(res: dict):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Risk of deterioration in {res['horizon_hours']}h", f"{res['risk_probability']:.1%}")
    c2.markdown(f"<div style='padding-top:8px'>Risk level<br><span style='font-size:28px;font-weight:700;"
                f"color:{LEVEL_COLOR[res['risk_level']]}'>{res['risk_level']}</span></div>", unsafe_allow_html=True)
    c3.metric("vs. average patient-hour", f"{res['relative_risk']:.1f}×")
    c4.metric("Alert threshold", f"{res['alert_threshold']:.1%}")
    box = {"HIGH": st.error, "MEDIUM": st.warning, "LOW": st.success}[res["risk_level"]]
    box("**AI summary** — " + res["summary"])


def factor_chart(res: dict):
    f = pd.DataFrame(res["top_factors"]).iloc[::-1]
    f["shown"] = [describe_factor(r) for r in f.to_dict("records")]
    fig = go.Figure(go.Bar(x=f["contribution"], y=f["shown"], orientation="h",
                           marker_color=np.where(f["contribution"] > 0, "#c62828", "#2e7d32")))
    fig.update_layout(title="What is driving this prediction (red raises risk, green lowers it)",
                      xaxis_title="Contribution to log-odds", height=320, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def forecast_table(res: dict, actual: pd.DataFrame | None = None) -> pd.DataFrame:
    """Rows = vitals, columns = Now, +1h ... +6h. Cells: 'predicted (low–high)', plus ' | actual X' if known."""
    rows = []
    for v, f in res["forecast"].items():
        row = {"Vital": f"{f['label']} ({f['unit']})", "Now": fmt(v, f["current"])}
        for p in f["path"]:
            cell = f"{fmt(v, p['predicted'])} ({fmt(v, p['low'])}–{fmt(v, p['high'])})"
            if actual is not None:
                a = actual.loc[actual["hour"] == res["hour_now"] + p["hours_ahead"], v]
                if len(a) and not pd.isna(a.iloc[0]):
                    hit = "✓" if p["low"] <= a.iloc[0] <= p["high"] else "✗"
                    cell += f" → real {fmt(v, a.iloc[0])} {hit}"
            row[f"+{p['hours_ahead']}h"] = cell
        rows.append(row)
    flags = {"Vital": "Danger-line flags", "Now": ""}
    for h in res["hourly_outlook"]:
        w = [("⚠️ " if x["level"] == "likely" else "· ") + VITAL_LABELS[x["vital"]] for x in h["warnings"]]
        flags[f"+{h['hours_ahead']}h"] = ", ".join(w) or "—"
    rows.append(flags)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------
st.title("🫀 ICU-PREDICT")
st.caption("6-hour deterioration risk and vital-sign forecasts · research prototype, not for clinical use")

try:
    pred = get_predictor()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

mode = st.sidebar.radio("View", ["Patient monitor", "Enter vitals manually", "Model performance"])
st.sidebar.markdown(f"**Model:** {pred.clf['model_name']}  \n**Horizon:** {pred.horizon} h  \n"
                    f"**Alert threshold:** {pred.threshold:.1%}")

if mode == "Patient monitor":
    cohort, timeline, labels, splits = load_data()
    test_ids = splits.loc[splits["split"] == "test", "stay_id"]
    test = cohort[cohort["stay_id"].isin(test_ids)].copy()
    ev = labels.groupby("stay_id")["event"].max()
    test["had_event"] = test["stay_id"].map(ev).fillna(0).astype(int)
    only_events = st.sidebar.checkbox("Only stays that deteriorated", value=True)
    if only_events:
        test = test[test["had_event"] == 1]
    test["label"] = test.apply(lambda r: f"{r['stay_id']} · {int(r['age'])}y {'M' if r['male'] else 'F'} · "
                                         f"{r['first_careunit'].split('(')[-1].rstrip(')')} · {r['n_hours']}h", axis=1)
    choice = st.sidebar.selectbox("ICU stay (held-out test patients)", test["label"])
    stay_id = int(choice.split(" · ")[0])
    srow = cohort[cohort["stay_id"] == stay_id].iloc[0]
    hist_all = timeline[timeline["stay_id"] == stay_id].sort_values("hour")
    max_h = int(hist_all["hour"].max())
    hour = st.slider("Current hour since ICU admission", 3, max_h, min(12, max_h))
    show_future = st.checkbox("Reveal what actually happened next (ground truth)", value=True)

    hist = hist_all[hist_all["hour"] <= hour]
    res = pred.predict(hist.drop(columns="stay_id"), srow.to_dict())
    show_result(res)

    traj = stay_trajectory(stay_id)
    lab = labels[labels["stay_id"] == stay_id]
    onsets = lab.loc[lab["event"] == 1, "hour"].tolist()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=traj["hour"][traj["hour"] <= hour], y=traj["risk"][traj["hour"] <= hour],
                             mode="lines", name="Predicted risk", line=dict(color="#1565c0", width=3)))
    if show_future:
        fig.add_trace(go.Scatter(x=traj["hour"][traj["hour"] >= hour], y=traj["risk"][traj["hour"] >= hour],
                                 mode="lines", name="Risk (future hours)", line=dict(color="#90caf9", dash="dot")))
        for o in onsets:
            fig.add_vline(x=o, line=dict(color="#c62828", dash="dash"))
        if onsets:
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color="#c62828", dash="dash"),
                                     name="Actual deterioration onset"))
    fig.add_hline(y=pred.threshold, line=dict(color="gray", dash="dot"), annotation_text="alert threshold")
    fig.add_vrect(x0=hour, x1=hour + pred.horizon, fillcolor="orange", opacity=0.12, line_width=0,
                  annotation_text="next 6h", annotation_position="top left")
    fig.update_layout(title="Deterioration risk over the stay", xaxis_title="Hour", yaxis_title="Risk",
                      yaxis_tickformat=".0%", height=330, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, width="stretch")

    vit = ["heart_rate", "sbp", "map", "resp_rate", "spo2", "temp_c"]
    th = CFG["labels"]["thresholds"]
    danger = {"heart_rate": th["hr_high"], "sbp": th["sbp_low"], "map": th["map_low"],
              "resp_rate": th["rr_high"], "spo2": th["spo2_low"]}
    fig = make_subplots(rows=2, cols=3, subplot_titles=[f"{VITAL_LABELS[v]} ({VITAL_UNITS[v]})" for v in vit])
    for i, v in enumerate(vit):
        r, c = i // 3 + 1, i % 3 + 1
        h = hist_all[hist_all["hour"] <= hour]
        fig.add_trace(go.Scatter(x=h["hour"], y=h[v], mode="lines+markers", marker=dict(size=3),
                                 line=dict(color="#1565c0"), showlegend=False), row=r, col=c)
        if show_future:
            f = hist_all[(hist_all["hour"] > hour) & (hist_all["hour"] <= hour + pred.horizon)]
            fig.add_trace(go.Scatter(x=f["hour"], y=f[v], mode="markers", marker=dict(size=4, color="#9e9e9e"),
                                     showlegend=False), row=r, col=c)
        fc = res["forecast"].get(v)
        if fc and fc["path"]:
            xs = [hour] + [hour + p["hours_ahead"] for p in fc["path"]]
            start = fc["current"] if fc["current"] is not None else fc["path"][0]["predicted"]
            mid = [start] + [p["predicted"] for p in fc["path"]]
            lo = [start] + [p["low"] for p in fc["path"]]
            hi = [start] + [p["high"] for p in fc["path"]]
            fig.add_trace(go.Scatter(x=xs + xs[::-1], y=hi + lo[::-1], fill="toself", fillcolor="rgba(239,108,0,0.18)",
                                     line=dict(width=0), hoverinfo="skip", showlegend=False), row=r, col=c)
            fig.add_trace(go.Scatter(x=xs, y=mid, mode="lines+markers", line=dict(color="#ef6c00", width=2.5),
                                     marker=dict(size=5), showlegend=False,
                                     hovertemplate="hour %{x}: %{y:.1f}<extra>forecast</extra>"), row=r, col=c)
        if v in danger:
            fig.add_hline(y=danger[v], line=dict(color="#c62828", dash="dot", width=1), row=r, col=c)
    fig.update_layout(height=520, title="Vitals so far (blue) · hour-by-hour forecast with 80% range (orange band)"
                      + (", what actually happened (grey)" if show_future else ""),
                      margin=dict(l=10, r=10, t=70, b=10))
    st.plotly_chart(fig, width="stretch")

    st.markdown(f"#### Hour by hour: ICU hours {hour + 1}–{hour + pred.horizon}")
    st.caption("Each cell: most likely value (80% range)"
               + (" → what really happened (✓ inside the range, ✗ outside)" if show_future else "")
               + ". ⚠️ = likely to cross a danger line · `·` = possible.")
    st.dataframe(forecast_table(res, hist_all if show_future else None), hide_index=True, width="stretch")
    st.plotly_chart(factor_chart(res), width="stretch")
    with st.expander("📋 Full plain-English report for the next 6 hours", expanded=True):
        st.markdown(res["detailed_report"])

elif mode == "Enter vitals manually":
    st.subheader("Score a patient from current and earlier vitals")
    st.caption("Enter vitals now and 6 hours ago; hours in between are interpolated. Leave labs at 0 if unknown.")
    with st.form("manual"):
        c1, c2, c3, c4 = st.columns(4)
        age = c1.number_input("Age", 18, 100, 67)
        male = c2.selectbox("Sex", ["Female", "Male"]) == "Male"
        unit = c3.selectbox("Care unit", ["Medical Intensive Care Unit (MICU)", "Surgical Intensive Care Unit (SICU)",
                                          "Cardiac Vascular Intensive Care Unit (CVICU)", "Coronary Care Unit (CCU)",
                                          "Trauma SICU (TSICU)"])
        vaso = c4.checkbox("On vasopressors now")
        fields = [("heart_rate", 88, 104), ("sbp", 126, 108), ("dbp", 70, 60), ("resp_rate", 17, 23),
                  ("spo2", 97, 94), ("temp_c", 37.0, 37.9)]
        st.markdown("**Vitals** (6 hours ago → now)")
        vals = {}
        cols = st.columns(len(fields))
        for col, (v, before, now) in zip(cols, fields):
            vals[v] = (col.number_input(f"{VITAL_LABELS.get(v, 'DBP')} 6h ago", value=float(before), key=v + "b"),
                       col.number_input(f"{VITAL_LABELS.get(v, 'DBP')} now", value=float(now), key=v + "n"))
        st.markdown("**Latest labs** (0 = not measured)")
        lc = st.columns(4)
        labs = {"lactate": lc[0].number_input("Lactate (mmol/L)", 0.0, 30.0, 2.1),
                "creatinine": lc[1].number_input("Creatinine (mg/dL)", 0.0, 20.0, 1.2),
                "wbc": lc[2].number_input("WBC (K/µL)", 0.0, 200.0, 12.0),
                "gcs_total": lc[3].number_input("GCS", 0.0, 15.0, 15.0)}
        go_btn = st.form_submit_button("Predict")
    if go_btn:
        n = 7
        hist = pd.DataFrame({v: np.linspace(b, a, n) for v, (b, a) in vals.items()})
        for k, v in labs.items():
            hist[k] = np.nan
            if v > 0:
                hist.loc[n - 1, k] = v
        hist["vasopressor"] = 0
        hist.loc[n - 1, "vasopressor"] = int(vaso)
        res = pred.predict(hist, {"age": age, "male": int(male), "emergency_admission": 1, "first_careunit": unit})
        show_result(res)
        st.dataframe(forecast_table(res), hide_index=True, width="stretch")
        st.plotly_chart(factor_chart(res), width="stretch")
        st.markdown(res["detailed_report"])

else:
    st.subheader("Held-out test-set performance")
    m = json.loads((OUT / "reports/metrics.json").read_text(encoding="utf-8"))
    t = m["test"]
    c = st.columns(5)
    c[0].metric("AUROC", f"{t['auroc']:.3f}", help=f"95% CI {m['auroc_ci'][0]:.3f}–{m['auroc_ci'][1]:.3f}")
    c[1].metric("AUPRC", f"{t['auprc']:.3f}", help=f"prevalence {t['prevalence']:.3f}")
    c[2].metric("Sensitivity", f"{t['sensitivity']:.1%}")
    c[3].metric("PPV", f"{t['ppv']:.1%}")
    c[4].metric("Alerts / patient-day", f"{t['alerts_per_patient_day']:.1f}")
    st.caption(f"Selected model: {m['selected_model']} · {m['data']['test_rows']:,} prediction points from "
               f"{m['data']['test_stays']} test stays · positive rate {t['prevalence']:.1%}")
    cvp = OUT / "reports/cv_results.csv"
    if cvp.exists():
        st.info("This cohort is small, so the single test set below is noisy. These patient-grouped "
                "cross-validation results use every stay and are the more reliable estimate.")
        st.dataframe(pd.read_csv(cvp), hide_index=True, width="stretch")
    comp = pd.read_csv(OUT / "reports/model_comparison.csv")
    st.dataframe(comp[["model", "auroc", "auprc", "brier", "ece", "sensitivity", "specificity", "ppv",
                       "episode_sensitivity", "median_lead_time_hours", "alerts_per_patient_day"]].round(3),
                 hide_index=True, width="stretch")
    for img in ["roc_pr_curves.png", "calibration_confusion.png", "feature_importance.png"]:
        p = OUT / "figures" / img
        if p.exists():
            st.image(str(p), width="stretch" if img != "feature_importance.png" else "content")
    fp = OUT / "reports/forecast_metrics.csv"
    if fp.exists():
        st.markdown("**6-hour vital forecasts vs. persistence baseline (MAE)**")
        st.dataframe(pd.read_csv(fp).round(3), hide_index=True, width="stretch")
