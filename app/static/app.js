/* ICU-PREDICT central station — talks to the FastAPI endpoints in app/api.py */
(() => {
  "use strict";

  const VITALS = [
    { key: "heart_rate", label: "Heart rate", short: "HR", unit: "bpm", cls: "hr", danger: ["above", "hr_high"] },
    { key: "sbp", label: "Systolic BP", short: "SBP", unit: "mmHg", cls: "sbp", danger: ["below", "sbp_low"] },
    { key: "map", label: "Mean arterial pressure", short: "MAP", unit: "mmHg", cls: "map", danger: ["below", "map_low"] },
    { key: "resp_rate", label: "Respiratory rate", short: "RR", unit: "/min", cls: "rr", danger: ["above", "rr_high"] },
    { key: "spo2", label: "Oxygen saturation", short: "SpO₂", unit: "%", cls: "spo2", danger: ["below", "spo2_low"] },
    { key: "temp_c", label: "Temperature", short: "Temp", unit: "°C", cls: "temp", danger: ["above", null, 38.3] },
  ];
  const DEC = { temp_c: 1 };

  const state = { t: 0, playing: false, timer: null, selected: null, reveal: false, cfg: null,
                  ward: null, patientKey: null, charts: [], busy: false };
  const $ = (s, el = document) => el.querySelector(s);
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const fmt = (k, v) => (v === null || v === undefined || Number.isNaN(v)) ? "–" : Number(v).toFixed(DEC[k] ?? 0);
  const pct = (p) => `${(p * 100).toFixed(p < 0.1 ? 1 : 0)}%`;
  const shortUnit = (u) => (u.match(/\(([^)]+)\)\s*$/) || [, u])[1];
  const levelClass = (l) => ({ HIGH: "high", MEDIUM: "medium", LOW: "low" }[l] || "low");

  function dangerLine(v) {
    const [dir, key, fixed] = v.danger;
    return { dir, value: key ? state.cfg.thresholds[key] : fixed };
  }

  async function api(path) {
    const r = await fetch(path);
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || `${r.status} ${r.statusText}`);
    return r.json();
  }

  // ---------------------------------------------------------------- ward board
  async function loadWard() {
    const data = await api(`/api/ward?t=${state.t}`);
    const prev = state.ward;
    state.ward = data;
    renderBoard(data.beds, prev);
    $("#simLabel").textContent = `Simulation hour ${state.t}`;
    if (state.selected) {
      const bed = data.beds.find((b) => b.stay_id === state.selected);
      if (bed) await loadPatient(bed);
    }
  }

  function sparkline(vals) {
    if (!vals.length) return "";
    const max = Math.max(0.25, ...vals), w = 84, h = 26;
    const pts = vals.map((v, i) => `${(i / Math.max(1, vals.length - 1)) * w},${h - 2 - (v / max) * (h - 4)}`).join(" ");
    const thrY = h - 2 - (state.cfg.alert_threshold / max) * (h - 4);
    return `<svg class="spark" viewBox="0 0 ${w} ${h}" aria-hidden="true">
      <line x1="0" x2="${w}" y1="${thrY}" y2="${thrY}" stroke="${css("--line")}" stroke-dasharray="2 2"/>
      <polyline points="${pts}"/></svg>`;
  }

  function renderBoard(beds, prev) {
    const nHigh = beds.filter((b) => b.level === "HIGH" && !b.end_of_record).length;
    const nMed = beds.filter((b) => b.level === "MEDIUM" && !b.end_of_record).length;
    $("#bedCount").textContent = `${beds.length} monitored`;
    $("#boardSummary").innerHTML =
      `<span class="count-pill high">${nHigh} early warning${nHigh === 1 ? "" : "s"}</span>` +
      `<span class="count-pill med">${nMed} to watch</span>` +
      `<span class="count-pill">${beds.length - nHigh - nMed} stable</span>`;

    // most urgent first, like a real central station
    const order = { HIGH: 0, MEDIUM: 1, LOW: 2 };
    const sorted = [...beds].sort((a, b) => (a.end_of_record - b.end_of_record) || (order[a.level] - order[b.level]) || (b.risk - a.risk));
    const wasHigh = new Set((prev?.beds || []).filter((b) => b.level === "HIGH").map((b) => b.stay_id));

    $("#beds").innerHTML = sorted.map((b) => {
      const v = b.vitals, lc = levelClass(b.level);
      const alarm = b.end_of_record ? `<span>Record ends at ICU hour ${b.max_hour}</span>`
        : b.level === "HIGH" ? `<span>Early warning</span><span>${pct(b.risk)} in 6 h</span>`
        : b.level === "MEDIUM" ? `<span>Watch</span><span>${pct(b.risk)} in 6 h</span>`
        : `<span>No early warning</span><span>${pct(b.risk)}</span>`;
      const flash = b.level === "HIGH" && !wasHigh.has(b.stay_id) && prev ? " flash" : "";
      const unit = shortUnit(b.unit);
      return `<button class="bed ${lc}${flash}${b.end_of_record ? " ended" : ""}" data-id="${b.stay_id}"
                aria-pressed="${state.selected === b.stay_id}" aria-label="Bed ${b.bed}, risk ${pct(b.risk)}">
        <div class="alarm">${alarm}</div>
        <div class="bed-body">
          <span class="bed-no">Bed ${b.bed}</span>${sparkline(b.risk_trend)}
          <span class="bed-pt">${b.age} ${b.sex} · ${esc(unit)} · ICU hour ${b.hour}${b.on_vasopressors ? " · on vasopressors" : ""}</span>
          <div class="bed-nums">
            <span class="bn hr"><b>${fmt("heart_rate", v.heart_rate)}</b><small>HR</small></span>
            <span class="bn bp"><b>${fmt("sbp", v.sbp)}/${fmt("dbp", v.dbp)}</b><small>BP</small></span>
            <span class="bn spo2"><b>${fmt("spo2", v.spo2)}</b><small>SpO₂</small></span>
            <span class="bn rr"><b>${fmt("resp_rate", v.resp_rate)}</b><small>RR</small></span>
          </div>
        </div>
      </button>`;
    }).join("");
    $("#beds").querySelectorAll(".bed").forEach((el) => el.addEventListener("click", () => selectBed(+el.dataset.id)));
  }

  function selectBed(id) {
    state.selected = id;
    state.patientKey = null;
    document.querySelectorAll(".bed").forEach((el) => el.setAttribute("aria-pressed", String(+el.dataset.id === id)));
    const bed = state.ward.beds.find((b) => b.stay_id === id);
    loadPatient(bed);
  }

  // ---------------------------------------------------------------- patient view
  async function loadPatient(bed) {
    const key = `${bed.stay_id}:${bed.hour}:${state.reveal}`;
    if (key === state.patientKey) return;
    state.patientKey = key;
    try {
      const r = await api(`/api/patient/${bed.stay_id}?hour=${bed.hour}&reveal=${state.reveal}`);
      if (state.patientKey === key) renderPatient(bed, r);
    } catch (e) {
      $("#detail").innerHTML = `<p class="error">Couldn't load bed ${esc(bed.bed)}: ${esc(e.message)}</p>`;
    }
  }

  function renderPatient(bed, r) {
    const d = $("#detail");
    state.charts.forEach((c) => c.destroy());
    state.charts = [];
    d.innerHTML = "";
    d.appendChild($("#detailTpl").content.cloneNode(true));
    d.classList.toggle("revealing", !!r.actual);
    const lc = levelClass(r.risk_level), p = r.patient;

    $(".pt-bed", d).textContent = `Bed ${bed.bed}`;
    $(".pt-line", d).textContent = `Stay ${p.stay_id} · ${p.age}-year-old ${p.sex === "M" ? "man" : "woman"} · ${p.unit}` +
      ` · ICU hour ${r.hour_now}${r.on_vasopressors ? " · on vasopressors" : ""}`;
    $("#printBtn", d).addEventListener("click", () => window.print());
    const menu = $("#dlMenu", d), btn = $("#dlBtn", d);
    btn.addEventListener("click", (e) => { e.stopPropagation(); menu.hidden = !menu.hidden; btn.setAttribute("aria-expanded", String(!menu.hidden)); });
    $("#dlPdf", d).addEventListener("click", (e) => {
      e.preventDefault(); menu.hidden = true;
      location.href = `/api/patient/${p.stay_id}/report.pdf?hour=${r.hour_now}&reveal=${!!r.actual}`;
    });
    $("#dlCsv", d).addEventListener("click", (e) => { e.preventDefault(); menu.hidden = true; downloadCsv(r, p); });

    // alarm banner
    const banner = $(".alarm-banner", d);
    banner.classList.add(lc);
    const head = r.risk_level === "HIGH" ? `<strong>Early warning.</strong> ${pct(r.risk_probability)} chance of deterioration in the next ${r.horizon_hours} hours.`
      : r.risk_level === "MEDIUM" ? `<strong>Watch.</strong> ${pct(r.risk_probability)} chance of deterioration in the next ${r.horizon_hours} hours.`
      : `<strong>No early warning.</strong> ${pct(r.risk_probability)} chance of deterioration in the next ${r.horizon_hours} hours.`;
    const caution = (r.currently_unstable || r.on_vasopressors)
      ? "<p>Already unstable or on vasopressors. The model is built to warn before deterioration, so read this score with caution.</p>" : "";
    let truth = "";
    if (r.actual) {
      const ev = r.actual.next_event_hour;
      const a = r.actual, H = r.horizon_hours;
      let what;
      if (ev && ev <= r.hour_now + H) what = `a new deterioration began at ICU hour ${ev}, ${ev - r.hour_now} h later.`;
      else if (a.unstable_hours > 0 || a.vasopressor_hours > 0) {
        const parts = [];
        if (a.unstable_hours) parts.push(`vital signs were past a danger line in ${a.unstable_hours} of the next ${H} hours`);
        if (a.vasopressor_hours) parts.push(`on vasopressors for ${a.vasopressor_hours} of them`);
        what = `${parts.join(", ")}. No new deterioration was counted because the patient was already unstable.`;
      } else what = `vital signs stayed on the safe side of every danger line for the next ${H} hours.`;
      truth = `<p>What happened: ${what}</p>`;
    }
    banner.innerHTML = head + caution + truth;

    renderNumerics(r, $(".numerics", d));
    renderRisk(r, d);
    renderDrivers(r, d);
    renderStrip(r, $(".strip", d));
    renderCharts(r, $(".charts", d));
    const md = r.detailed_report.split("\n").filter((l, i) => !(i < 2 && (l.startsWith("## ") || l.startsWith("*ICU")))).join("\n");
    $("#report", d).innerHTML = window.marked ? marked.parse(md) : `<pre>${esc(md)}</pre>`;
  }

  function worstFlag(r, key) {
    let worst = null;
    for (const h of r.hourly_outlook) for (const w of h.warnings) if (w.vital === key) {
      if (w.level === "likely") return { level: "likely", h: h.hours_ahead };
      if (!worst) worst = { level: "possible", h: h.hours_ahead };
    }
    return worst;
  }

  function renderNumerics(r, el) {
    el.innerHTML = VITALS.map((v) => {
      const f = r.forecast[v.key];
      if (!f) return "";
      const ch = f.change_last_6h, hrs = f.change_hours ?? 6;
      const trend = ch === null || ch === undefined ? "No recent trend"
        : `${ch > 0 ? "▲" : ch < 0 ? "▼" : "■"} ${fmt(v.key, Math.abs(ch))} over ${hrs} h`;
      const last = f.path[f.path.length - 1];
      const flag = worstFlag(r, v.key);
      const fcls = flag ? (flag.level === "likely" ? "danger" : "warn") : "";
      const ftxt = flag ? (flag.level === "likely" ? `Likely crosses by +${flag.h} h` : `May cross by +${flag.h} h`) : "";
      return `<div class="vital ${fcls}">
        ${flag ? `<span class="flag">${ftxt}</span>` : ""}
        <div class="vital-top"><span>${v.short}</span><span>${v.unit}</span></div>
        <div class="val c-${v.cls}">${fmt(v.key, f.current)}</div>
        <div class="trend">${trend}</div>
        <div class="fc">In ${last.hours_ahead} h <b class="c-${v.cls}">${fmt(v.key, last.predicted)}</b>
          <span class="muted">(${fmt(v.key, last.low)}–${fmt(v.key, last.high)})</span></div>
      </div>`;
    }).join("");
  }

  function renderRisk(r, d) {
    const lc = levelClass(r.risk_level);
    const n = $(".risk-num", d);
    n.textContent = pct(r.risk_probability);
    n.classList.add(lc);
    const per100 = Math.round(r.risk_probability * 100), base = Math.round(r.baseline_risk * 100);
    $(".risk-context", d).innerHTML = `About <b>${per100} in 100</b> patients who look like this deteriorate within
      ${r.horizon_hours} hours, against ${base} in 100 for an average ICU hour (<b>${r.relative_risk.toFixed(1)}×</b>).
      Alerts start at ${pct(r.alert_threshold)}.`;
    const rh = r.risk_history;
    const ctx = $("#riskChart", d);
    state.charts.push(new Chart(ctx, {
      type: "line",
      data: { datasets: [
        { data: rh.hour.map((h, i) => ({ x: h, y: rh.risk[i] * 100 })), borderColor: css("--text"), borderWidth: 2, pointRadius: 0, tension: 0.2 },
        { data: [{ x: rh.hour[0], y: r.alert_threshold * 100 }, { x: rh.hour[rh.hour.length - 1], y: r.alert_threshold * 100 }],
          borderColor: css("--alarm-high"), borderDash: [3, 3], borderWidth: 1.2, pointRadius: 0 },
      ] },
      options: baseOptions({ yTitle: "%", xMin: rh.hour[0], xMax: rh.hour[rh.hour.length - 1], yMin: 0 }),
    }));
  }

  function renderDrivers(r, d) {
    const max = Math.max(...r.top_factors.map((f) => Math.abs(f.contribution)), 1e-6);
    $(".drivers", d).innerHTML = r.top_factors.map((f) => {
      const w = (Math.abs(f.contribution) / max) * 50, up = f.contribution > 0;
      const style = up ? `left:50%;width:${w}%` : `left:${50 - w}%;width:${w}%`;
      return `<li><span>${esc(f.text)}<br><span class="dir">${up ? "Raises risk" : "Lowers risk"}</span></span>
        <span class="dbar" aria-hidden="true"><i class="${up ? "up" : "down"}" style="${style}"></i></span></li>`;
    }).join("");
    const missing = r.top_factors.some((f) => f.value === null || f.feature.endsWith("_hrs_since"));
    $(".drivers-note", d).textContent = missing
      ? "Tests that haven't been ordered count too: clinicians order them when they're worried." : "";
  }

  function renderStrip(r, el) {
    const hours = r.hourly_outlook;
    let html = `<div class="sh">Hours ahead</div>` + hours.map((h) =>
      `<div class="sh"><b>+${h.hours_ahead} h</b>ICU hour ${h.icu_hour}</div>`).join("");
    for (const v of VITALS) {
      const f = r.forecast[v.key];
      if (!f) continue;
      html += `<div class="sl">${v.short} <span class="muted">${v.unit}</span></div>`;
      for (const h of hours) {
        const p = h.vitals[v.key];
        if (!p) { html += "<div>–</div>"; continue; }
        const w = h.warnings.find((x) => x.vital === v.key);
        let actual = "";
        if (r.actual) {
          const i = r.actual.hour.indexOf(h.icu_hour), a = i >= 0 ? r.actual[v.key][i] : null;
          if (a !== null && a !== undefined) {
            const inside = a >= p.low && a <= p.high;
            actual = `<span class="actual ${inside ? "in" : "out"}">Real ${fmt(v.key, a)} ${inside ? "✓" : "✗"}</span>`;
          }
        }
        html += `<div class="sv ${w ? w.level : ""} c-wrap"><b class="c-${v.cls}">${fmt(v.key, p.predicted)}</b>
          <span>${fmt(v.key, p.low)}–${fmt(v.key, p.high)}${w ? (w.level === "likely" ? " · likely past line" : " · may reach line") : ""}</span>${actual}</div>`;
      }
    }
    el.innerHTML = html;
  }

  function baseOptions({ yTitle, xMin, xMax, yMin, yMax }) {
    const grid = css("--line"), muted = css("--muted");
    return {
      animation: false, responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: { enabled: true, intersect: false, mode: "nearest" } },
      scales: {
        x: { type: "linear", min: xMin, max: xMax, grid: { color: grid }, ticks: { color: muted, stepSize: 6, font: { size: 11 } },
             title: { display: false } },
        y: { min: yMin, max: yMax, grid: { color: grid }, ticks: { color: muted, font: { size: 11 }, maxTicksLimit: 5 },
             title: { display: !!yTitle, text: yTitle, color: muted, font: { size: 11 } } },
      },
      elements: { line: { spanGaps: true } },
    };
  }

  function renderCharts(r, el) {
    el.innerHTML = VITALS.filter((v) => r.forecast[v.key]).map((v) =>
      `<div class="chart-card"><h4 class="c-${v.cls}">${v.label} <span class="muted">${v.unit}</span></h4>
        <div class="chart-wrap"><canvas id="ch-${v.key}" role="img" aria-label="${v.label} trend and forecast"></canvas></div></div>`).join("");
    const H = r.history, now = r.hour_now, end = now + r.horizon_hours;
    for (const v of VITALS) {
      const f = r.forecast[v.key];
      if (!f) continue;
      const color = css(`--${v.cls === "sbp" ? "bp" : v.cls}`);
      const hist = H.hour.map((h, i) => ({ x: h, y: H[v.key][i] })).filter((p) => p.y !== null);
      const start = f.current ?? f.path[0].predicted;
      const xs = [now, ...f.path.map((p) => now + p.hours_ahead)];
      const lo = [start, ...f.path.map((p) => p.low)], hi = [start, ...f.path.map((p) => p.high)];
      const md = [start, ...f.path.map((p) => p.predicted)];
      const dl = dangerLine(v);
      const xMin = Math.max(0, now - 23);
      const ds = [
        { data: xs.map((x, i) => ({ x, y: lo[i] })), borderWidth: 0, pointRadius: 0 },
        { data: xs.map((x, i) => ({ x, y: hi[i] })), borderWidth: 0, pointRadius: 0, fill: "-1", backgroundColor: hexA(color, 0.18) },
        { data: xs.map((x, i) => ({ x, y: md[i] })), borderColor: color, borderDash: [5, 4], borderWidth: 2, pointRadius: 2, pointBackgroundColor: color },
        { data: hist, borderColor: color, borderWidth: 2, pointRadius: 1.5, pointBackgroundColor: color, tension: 0.15 },
        { data: [{ x: xMin, y: dl.value }, { x: end, y: dl.value }], borderColor: css("--alarm-high"), borderDash: [2, 3], borderWidth: 1.2, pointRadius: 0 },
      ];
      if (r.actual) {
        ds.push({ type: "scatter", data: r.actual.hour.map((h, i) => ({ x: h, y: r.actual[v.key][i] })).filter((p) => p.y !== null),
                  pointRadius: 3.5, pointBackgroundColor: css("--text"), borderColor: css("--text") });
      }
      const ys = [...hist.map((p) => p.y), ...lo, ...hi, dl.value, ...(r.actual ? r.actual[v.key].filter((y) => y !== null) : [])];
      const pad = (Math.max(...ys) - Math.min(...ys)) * 0.08 || 1;
      state.charts.push(new Chart($(`#ch-${v.key}`), {
        type: "line", data: { datasets: ds },
        options: { ...baseOptions({ xMin, xMax: end, yMin: Math.floor(Math.min(...ys) - pad), yMax: Math.ceil(Math.max(...ys) + pad) }),
                   plugins: { legend: { display: false }, tooltip: { enabled: false } } },
        plugins: [nowLine(now)],
      }));
    }
  }

  const nowLine = (now) => ({
    id: "now",
    afterDatasetsDraw(chart) {
      const x = chart.scales.x.getPixelForValue(now), { top, bottom } = chart.chartArea, c = chart.ctx;
      c.save(); c.strokeStyle = css("--muted"); c.setLineDash([]); c.lineWidth = 1;
      c.beginPath(); c.moveTo(x, top); c.lineTo(x, bottom); c.stroke(); c.restore();
    },
  });

  function hexA(hex, a) {
    const h = hex.replace("#", ""), n = parseInt(h, 16);
    return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
  }

  function downloadCsv(r, p) {
    const keys = VITALS.map((v) => v.key);
    const head = ["icu_hour", "type", ...keys.flatMap((k) => [k, `${k}_low`, `${k}_high`]), "risk_pct_next_6h"];
    const riskAt = Object.fromEntries(r.risk_history.hour.map((h, i) => [h, r.risk_history.risk[i]]));
    const rows = [head];
    r.history.hour.forEach((h, i) => rows.push([h, "recorded", ...keys.flatMap((k) => [r.history[k][i] ?? "", "", ""]),
      riskAt[h] !== undefined ? (riskAt[h] * 100).toFixed(2) : ""]));
    r.hourly_outlook.forEach((h) => rows.push([h.icu_hour, "forecast",
      ...keys.flatMap((k) => { const q = h.vitals[k]; return q ? [q.predicted.toFixed(1), q.low.toFixed(1), q.high.toFixed(1)] : ["", "", ""]; }), ""]));
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([rows.map((row) => row.join(",")).join("\n")], { type: "text/csv" }));
    a.download = `icu-data_stay-${p.stay_id}_hour-${r.hour_now}.csv`;
    document.body.appendChild(a); a.click(); a.remove();
  }

  // ---------------------------------------------------------------- simulation clock
  async function tick(delta) {
    if (state.busy) return;
    state.busy = true;
    state.t = Math.max(0, state.t + delta);
    try { await loadWard(); } catch (e) { console.error(e); } finally { state.busy = false; }
    if (state.ward && state.ward.beds.every((b) => b.end_of_record) && state.playing) togglePlay();
  }

  function togglePlay() {
    state.playing = !state.playing;
    $("#playBtn").setAttribute("aria-pressed", String(state.playing));
    $("#playLabel").textContent = state.playing ? "Pause" : "Play";
    $("#liveDot").classList.toggle("on", state.playing);
    clearInterval(state.timer);
    if (state.playing) state.timer = setInterval(() => tick(1), +$("#speed").value);
  }

  async function init() {
    document.addEventListener("click", () => document.querySelectorAll(".dl-menu").forEach((m) => (m.hidden = true)));
    $("#playBtn").addEventListener("click", togglePlay);
    $("#stepFwd").addEventListener("click", () => tick(1));
    $("#stepBack").addEventListener("click", () => tick(-1));
    $("#speed").addEventListener("change", () => { if (state.playing) { togglePlay(); togglePlay(); } });
    $("#reveal").addEventListener("change", (e) => {
      state.reveal = e.target.checked;
      state.patientKey = null;
      const bed = state.ward?.beds.find((b) => b.stay_id === state.selected);
      if (bed) loadPatient(bed);
    });
    if (window.Chart) Chart.defaults.font.family = css("--ui");
    try {
      state.cfg = await api("/api/config");
      const src = {
        "mimic-iv-demo": "De-identified patients from the MIMIC-IV Clinical Database Demo v2.2 (Johnson et al., PhysioNet, ODbL).",
        "mimic-iv": "De-identified patients from MIMIC-IV (PhysioNet). For credentialed local use only.",
        synthetic: "Synthetic patients generated in the MIMIC-IV format; they are not real people.",
      }[state.cfg.data_source] || "";
      $("#foot").textContent = `${src} Predictions are from a research model and must not be used for patient care.`;
      await loadWard();
      const first = [...state.ward.beds].sort((a, b) => b.risk - a.risk)[0];
      if (first) selectBed(first.stay_id);
    } catch (e) {
      $("#beds").innerHTML = `<p class="error">Couldn't load the ward: ${esc(e.message)}. Run the pipeline first
        (python run_pipeline.py --demo or --mimic-dir …), then restart the server.</p>`;
    }
  }
  init();
})();
