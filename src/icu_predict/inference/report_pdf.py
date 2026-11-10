"""Downloadable PDF version of a patient's 6-hour report (charts + tables + written summary)."""
from __future__ import annotations

import io
import re
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

from icu_predict.inference.predictor import VITAL_LABELS, VITAL_UNITS, danger_lines, fmt

INK, MUTED, LINE = colors.HexColor("#1F2937"), colors.HexColor("#6B7280"), colors.HexColor("#E5E7EB")
LEVEL_COLOR = {"HIGH": "#D92D20", "MEDIUM": "#E8A317", "LOW": "#0E9F4F"}
VITAL_COLOR = {"heart_rate": "#0E9F6E", "sbp": "#D92D20", "map": "#E46A76", "resp_rate": "#C08A00",
               "spo2": "#1C7ED6", "temp_c": "#7C5CC4"}


def _styles():
    ss = getSampleStyleSheet()
    base = ParagraphStyle("base", parent=ss["Normal"], fontName="Helvetica", fontSize=9.5, leading=13.5,
                          textColor=INK, alignment=TA_LEFT)
    return {
        "base": base,
        "title": ParagraphStyle("title", parent=base, fontName="Helvetica-Bold", fontSize=17, leading=21),
        "sub": ParagraphStyle("sub", parent=base, textColor=MUTED, fontSize=9),
        "h2": ParagraphStyle("h2", parent=base, fontName="Helvetica-Bold", fontSize=11.5, leading=15,
                             spaceBefore=10, spaceAfter=4),
        "bullet": ParagraphStyle("bullet", parent=base, leftIndent=10, bulletIndent=0, spaceAfter=2),
        "note": ParagraphStyle("note", parent=base, backColor=colors.HexColor("#FFF7E6"), borderPadding=6,
                               leftIndent=6, rightIndent=6, spaceBefore=6, spaceAfter=6),
        "small": ParagraphStyle("small", parent=base, fontSize=8, leading=11, textColor=MUTED),
        "big": ParagraphStyle("big", parent=base, fontName="Helvetica-Bold", fontSize=26, leading=30),
    }


def _md_inline(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)", r"<i>\1</i>", text)
    text = text.replace("`", "")
    return text.replace("⚠️", "!").replace("↑", "").replace("↓", "").replace(" → ", " to ").replace("→", "to")


def _report_section(md: str, number: int) -> list[str]:
    m = re.search(rf"### {number}\. .*?\n(.*?)(?=\n### |\Z)", md, flags=re.S)
    return [l for l in (m.group(1).strip().splitlines() if m else []) if l.strip()]


def _vitals_chart(res: dict, history: dict, actual: dict | None) -> io.BytesIO:
    th = danger_lines(res.get("_thresholds", {})) if res.get("_thresholds") else {}
    vit = [v for v in ["heart_rate", "sbp", "map", "resp_rate", "spo2", "temp_c"] if v in res["forecast"]]
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.2))
    now = res["hour_now"]
    for ax, v in zip(axes.flat, vit):
        f, c = res["forecast"][v], VITAL_COLOR[v]
        hx = [h for h, y in zip(history["hour"], history[v]) if y is not None]
        hy = [y for y in history[v] if y is not None]
        xs = [now] + [now + p["hours_ahead"] for p in f["path"]]
        start = f["current"] if f["current"] is not None else f["path"][0]["predicted"]
        ax.fill_between(xs, [start] + [p["low"] for p in f["path"]], [start] + [p["high"] for p in f["path"]],
                        color=c, alpha=0.15, lw=0)
        ax.plot(xs, [start] + [p["predicted"] for p in f["path"]], "--o", color=c, ms=2.5, lw=1.4)
        ax.plot(hx, hy, "-o", color=c, ms=2, lw=1.6)
        if actual:
            ax.plot([h for h, y in zip(actual["hour"], actual[v]) if y is not None],
                    [y for y in actual[v] if y is not None], "o", color="#111827", ms=3)
        if v in th:
            ax.axhline(th[v][1], color="#D92D20", ls=":", lw=1)
        ax.axvline(now, color="#9CA3AF", lw=0.8)
        ax.set_title(f"{VITAL_LABELS[v]} ({VITAL_UNITS[v]})", fontsize=9, loc="left", color="#1F2937")
        ax.tick_params(labelsize=7, colors="#6B7280")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(alpha=0.25)
    for ax in list(axes.flat)[len(vit):]:
        ax.axis("off")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=170)
    plt.close(fig)
    buf.seek(0)
    return buf


def _risk_chart(res: dict) -> io.BytesIO:
    rh = res["risk_history"]
    hrs, risk = rh["hour"][-24:], [r * 100 for r in rh["risk"][-24:]]
    thr = res["alert_threshold"] * 100
    col = ["#D92D20" if r >= thr else "#E8A317" if r >= thr / 2 else "#0E9F4F" for r in risk]
    fig, ax = plt.subplots(figsize=(10, 2.2))
    ax.bar(hrs, risk, color=col, width=0.75)
    ax.axhline(thr, color="#E8A317", lw=1.5, ls="--")
    ax.text(hrs[0] - 0.4, thr, f" alert threshold {thr:.1f}%", va="bottom", fontsize=7, color="#92400E")
    ax.set_ylabel("Risk (%)", fontsize=8, color="#6B7280")
    ax.set_xlabel("ICU hour", fontsize=8, color="#6B7280")
    ax.tick_params(labelsize=7, colors="#6B7280")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=170)
    plt.close(fig)
    buf.seek(0)
    return buf


def build_pdf(res: dict, thresholds: dict) -> bytes:
    """res: output of the /api/patient endpoint (prediction + history + risk_history + patient)."""
    st = _styles()
    res = {**res, "_thresholds": thresholds}
    p, H = res["patient"], res["horizon_hours"]
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm, topMargin=14 * mm,
                            bottomMargin=14 * mm, title=f"ICU-PREDICT report - stay {p['stay_id']}",
                            author="ICU-PREDICT")
    W = doc.width
    story = [
        Paragraph("ICU Patient Report: next 6 hours", st["title"]),
        Paragraph(f"Stay {p['stay_id']} · {p['age']}-year-old {'man' if p['sex'] == 'M' else 'woman'} · "
                  f"{_md_inline(p['unit'])} · ICU hour {res['hour_now']} · generated "
                  f"{datetime.now().strftime('%b %d, %Y %I:%M %p')}", st["sub"]),
        Spacer(1, 8),
    ]

    lvl = res["risk_level"]
    risk_cell = [Paragraph(f"<font color='{LEVEL_COLOR[lvl]}'>{res['risk_probability'] * 100:.0f}%</font>", st["big"]),
                 Paragraph(f"6-hour deterioration risk · <b>{lvl.title()}</b>"
                           + (" · alert" if res["alert"] else ""), st["sub"])]
    words = Paragraph(
        f"About <b>{res['risk_probability'] * 100:.0f} in 100</b> patients who look like this deteriorate within "
        f"{H} hours, against {res['baseline_risk'] * 100:.0f} in 100 for an average ICU hour "
        f"(<b>{res['relative_risk']:.1f}x</b>). Deterioration means a vasopressor is started, the patient dies, "
        "or vital signs stay past a danger line for 2+ hours.", st["base"])
    t = Table([[risk_cell, words]], colWidths=[48 * mm, W - 48 * mm])
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("BOX", (0, 0), (-1, -1), 0.6, LINE),
                           ("BACKGROUND", (0, 0), (0, 0), colors.HexColor("#F9FAFB")),
                           ("LEFTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 8),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
    story += [t, Spacer(1, 6)]
    if res.get("currently_unstable") or res.get("on_vasopressors"):
        story.append(Paragraph("<b>Caution:</b> the patient is already unstable or on vasopressors. The model is "
                               "built to warn before deterioration, so this score is less reliable here.", st["note"]))

    # current vitals + 6h forecast table
    rows = [["Vital", "Now", "Change", f"In {H} h (80% range)"]]
    for v, f in res["forecast"].items():
        ch = f.get("change_last_6h")
        rows.append([f"{f['label']} ({f['unit']})", fmt(v, f["current"]),
                     "–" if ch is None else f"{'+' if ch > 0 else ''}{fmt(v, ch)} / {f.get('change_hours', 6)} h",
                     f"{fmt(v, f['predicted'])} ({fmt(v, f['low'])}–{fmt(v, f['high'])})"])
    story += [Paragraph("Where the patient is now", st["h2"]), _table(rows, W, [0.36, 0.14, 0.2, 0.3])]

    story += [Paragraph("Vital trends and hour-by-hour forecast", st["h2"]),
              Image(_vitals_chart(res, res["history"], res.get("actual")), width=W, height=W * 0.5),
              Paragraph("Solid line: recorded. Dashed line and shading: forecast with its 80% range. Dotted red: "
                        "danger line." + (" Black dots: what actually happened." if res.get("actual") else ""),
                        st["small"])]

    hrs = [h["hours_ahead"] for h in res["hourly_outlook"]]
    rows = [["Vital"] + [f"+{k} h" for k in hrs]]
    for v, f in res["forecast"].items():
        rows.append([f["label"]] + [f"{fmt(v, q['predicted'])}\n{fmt(v, q['low'])}–{fmt(v, q['high'])}" for q in f["path"]])
    short = {"heart_rate": "HR", "sbp": "SBP", "map": "MAP", "resp_rate": "RR", "spo2": "SpO2", "temp_c": "Temp"}
    cell = ParagraphStyle("cell", parent=st["small"], alignment=2, fontSize=7.5, leading=9.5)
    flags = [Paragraph("Danger-line flags<br/><font size=6.5>likely / may reach</font>", st["small"])]
    for h in res["hourly_outlook"]:
        likely = [short[w["vital"]] for w in h["warnings"] if w["level"] == "likely"]
        maybe = [short[w["vital"]] for w in h["warnings"] if w["level"] != "likely"]
        txt = "<br/>".join(([f"<font color='#D92D20'><b>{', '.join(likely)}</b></font>"] if likely else [])
                           + ([", ".join(maybe)] if maybe else [])) or "–"
        flags.append(Paragraph(txt, cell))
    rows.append(flags)
    story += [KeepTogether([Paragraph("Next 6 hours, hour by hour", st["h2"]),
                            _table(rows, W, [0.22] + [0.13] * len(hrs), font=7.5)])]

    story += [KeepTogether([Paragraph("Deterioration risk, last 24 hours", st["h2"]),
                            Image(_risk_chart(res), width=W, height=W * 0.22)])]

    md = res["detailed_report"]
    for num, title in ((4, "What to watch"), (5, "Why the model gives this risk"), (6, "How much to trust this")):
        lines = _report_section(md, num)
        if not lines:
            continue
        story.append(Paragraph(title, st["h2"]))
        for l in lines:
            if l.startswith("- "):
                story.append(Paragraph(_md_inline(l[2:]), st["bullet"], bulletText="•"))
            elif l.startswith(">"):
                story.append(Paragraph(_md_inline(l.lstrip("> ")), st["note"]))
            else:
                story.append(Paragraph(_md_inline(l), st["base"]))

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buf.getvalue()


def _table(rows, width, fractions, font=8.5):
    t = Table(rows, colWidths=[width * f for f in fractions], repeatRows=1)
    t.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), "Helvetica", font), ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", font),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK), ("TEXTCOLOR", (0, 0), (-1, 0), MUTED),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"), ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(16 * mm, 8 * mm, "ICU-PREDICT research prototype, not for clinical decisions. "
                      "Data: MIMIC-IV Clinical Database Demo (PhysioNet, ODbL).")
    canvas.drawRightString(A4[0] - 16 * mm, 8 * mm, f"Page {doc.page}")
    canvas.restoreState()
