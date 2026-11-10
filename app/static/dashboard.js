/* ICU-PREDICT patient dashboard */
(() => {
  "use strict";
  const $ = (s) => document.querySelector(s);
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const num = (v, d = 0) => (v === null || v === undefined || Number.isNaN(v)) ? "–" : Number(v).toFixed(d);
  const pct = (p) => `${(p * 100).toFixed(p < 0.1 ? 1 : 0)}%`;

  const ICONS = {
    temp: '<path d="M14 14.8V5a2 2 0 0 0-4 0v9.8a4 4 0 1 0 4 0z"/><path d="M12 9v7"/><path d="M17.5 6h3M17.5 9h3M17.5 12h2"/>',
    pulse: '<path d="M2 12h4.5l2-5 3.5 11 2.5-6H22"/>',
    lungs: '<path d="M12 3v8m0 0c-.8 2-2.6 3-4.6 3M12 11c.8 2 2.6 3 4.6 3"/><path d="M8.4 7C5 8 3 12.5 3 17c0 1.8 1 3 2.6 3C8.5 20 9.5 18 9.5 15V8.3M15.6 7C19 8 21 12.5 21 17c0 1.8-1 3-2.6 3-2.9 0-3.9-2-3.9-5V8.3"/>',
    spo2: '<rect x="7.5" y="2.5" width="9" height="10.5" rx="2"/><path d="M9.5 13v5.5a2.5 2.5 0 0 0 5 0V13"/><path d="M10 6.5h4M12 4.5v4"/>',
    flask: '<path d="M9 3h6M10 3v6.2L4.6 18.6A1.6 1.6 0 0 0 6 21h12a1.6 1.6 0 0 0 1.4-2.4L14 9.2V3"/><path d="M7 15.5h10"/>',
    drop: '<path d="M12 3c3.2 4.2 6 7.6 6 11.2A6 6 0 0 1 6 14.2C6 10.6 8.8 7.2 12 3z"/><path d="M12 11.5v5M9.5 14h5"/>',
    kidney: '<path d="M9 3.5C5.4 3.5 3 7 3 11.5S5 20.5 9 20.5c2.2 0 3-1.8 3-3.6 0-2.2-2.2-2.6-2.2-5.4S12 8.2 12 6.2 10.7 3.5 9 3.5z"/><path d="M15 4c3.2.8 6 3.8 6 7.8 0 4.6-2.6 8.2-6 8.7"/>',
  };
  const TILES = [
    { key: "temp_c", label: "Body temperature", unit: "°C", dec: 1, icon: "temp",
      status: (v) => v > 38.3 || v < 35 ? "bad" : v >= 38 || v < 36 ? "warn" : "ok" },
    { key: "heart_rate", label: "Heart rate", unit: "bpm", icon: "pulse",
      status: (v) => v > 130 || v < 45 ? "bad" : v > 100 || v < 55 ? "warn" : "ok" },
    { key: "resp_rate", label: "Respiratory rate", unit: "breaths/min", icon: "lungs",
      status: (v) => v > 30 || v < 8 ? "bad" : v > 20 || v < 10 ? "warn" : "ok" },
    { key: "spo2", label: "Oxygen saturation", unit: "%", icon: "spo2",
      status: (v) => v < 88 ? "bad" : v < 94 ? "warn" : "ok" },
    { key: "lactate", label: "Lactate", unit: "mmol/L", dec: 1, icon: "flask", lab: true,
      status: (v) => v > 4 ? "bad" : v >= 2 ? "warn" : "ok" },
    { key: "glucose", label: "Blood glucose", unit: "mg/dL", icon: "drop", lab: true,
      status: (v) => v < 60 || v > 300 ? "bad" : v < 70 || v > 180 ? "warn" : "ok" },
    { key: "creatinine", label: "Creatinine", unit: "mg/dL", dec: 2, icon: "kidney", lab: true,
      status: (v) => v > 2 ? "bad" : v > 1.3 ? "warn" : "ok" },
  ];
  const VITALS = [
    { key: "heart_rate", label: "Heart rate", unit: "bpm", color: "--hr", danger: ["above", "hr_high"] },
    { key: "sbp", label: "Systolic BP", unit: "mmHg", color: "--sbp", danger: ["below", "sbp_low"] },
    { key: "map", label: "Mean arterial pressure", tab: "MAP", unit: "mmHg", color: "--map", danger: ["below", "map_low"] },
    { key: "resp_rate", label: "Respiratory rate", tab: "Resp. rate", unit: "/min", color: "--rr", danger: ["above", "rr_high"] },
    { key: "spo2", label: "Oxygen saturation", tab: "SpO₂", unit: "%", color: "--spo2", danger: ["below", "spo2_low"] },
    { key: "temp_c", label: "Temperature", unit: "°C", dec: 1, color: "--temp", danger: ["above", null, 38.3] },
  ];

  const state = { patients: [], stay: null, hour: 24, vital: "heart_rate", reveal: false, res: null, cfg: null, charts: {} };

  async function api(path) {
    const r = await fetch(path);
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || `${r.status} ${r.statusText}`);
    return r.json();
  }

  // ---------------------------------------------------------------- gauges (SVG)
  function gauge({ min, max, value, zones, ticks, text, tickFmt = (t) => t }) {
    const cx = 120, cy = 118, r = 92, w = 30;
    const ang = (v) => Math.PI * (1 - (Math.min(max, Math.max(min, v)) - min) / (max - min));
    const pt = (a, rr) => [cx + rr * Math.cos(a), cy - rr * Math.sin(a)];
    const arc = (v0, v1, color) => {
      const [x0, y0] = pt(ang(v0), r), [x1, y1] = pt(ang(v1), r);
      return `<path d="M${x0.toFixed(2)} ${y0.toFixed(2)} A${r} ${r} 0 0 1 ${x1.toFixed(2)} ${y1.toFixed(2)}" stroke="${color}" />`;
    };
    let from = min, segs = "";
    for (const z of zones) { if (z.to > from) segs += arc(from, z.to, z.color); from = z.to; }
    const tk = ticks.map((t) => {
      const [x, y] = pt(ang(t), r - w / 2 - 16);
      return `<text x="${x.toFixed(1)}" y="${(y + 4).toFixed(1)}" text-anchor="middle" font-size="11.5" fill="${css("--ink-2")}">${tickFmt(t)}</text>`;
    }).join("");
    const a = ang(value ?? min), [nx, ny] = pt(a, r - 8);
    const [b1x, b1y] = pt(a + Math.PI / 2, 7), [b2x, b2y] = pt(a - Math.PI / 2, 7);
    const needle = value === null || value === undefined ? "" :
      `<path d="M${b1x.toFixed(1)} ${b1y.toFixed(1)} L${nx.toFixed(1)} ${ny.toFixed(1)} L${b2x.toFixed(1)} ${b2y.toFixed(1)} Z" fill="#475467"/>
       <circle cx="${cx}" cy="${cy}" r="8" fill="#475467"/>`;
    return `<svg class="gauge" viewBox="0 0 240 160" role="img" aria-label="${esc(text)}">
      <g fill="none" stroke-width="${w}">${segs}</g>${tk}${needle}
      <text x="${cx}" y="152" text-anchor="middle" font-size="20" fill="${css("--ink-2")}">${esc(text)}</text></svg>`;
  }

  // ---------------------------------------------------------------- render
  function render() {
    const r = state.res, p = r.patient, thr = r.alert_threshold;
    const pidText = `Stay ${p.stay_id} · ${p.age} ${p.sex} · ICU hour ${r.hour_now}`;
    ["#pid1", "#pid2", "#pid3"].forEach((s) => ($(s).textContent = pidText));
    $("#refreshed").textContent = `Data refreshed at ${new Date().toLocaleString(undefined, { month: "short", day: "numeric", year: "numeric", hour: "numeric", minute: "2-digit" })}`;
    $("#hourInput").value = r.hour_now;
    $("#hourMax").textContent = `of ${p.max_hour}`;
    $("#pdfBtn").href = pdfUrl();

    renderTiles(r);
    renderGauges(r, thr);
    renderBanner(r);
    renderLine(r);
    renderRiskBars(r, thr);
    renderReport(r);
  }

  function lastLab(r, key) {
    const h = r.history.hour, v = r.history[key] || [];
    for (let i = v.length - 1; i >= 0; i--) if (v[i] !== null && v[i] !== undefined) return { value: v[i], ago: r.hour_now - h[i] };
    return null;
  }

  function renderTiles(r) {
    $("#tileSlots").innerHTML = TILES.map((t) => {
      let value = null, extra = "";
      if (t.lab) {
        const l = lastLab(r, t.key);
        if (l) { value = l.value; extra = l.ago === 0 ? "Measured this hour" : `Measured ${l.ago} h ago`; }
        else extra = "Not measured in the last 24 h";
      } else {
        const f = r.forecast[t.key];
        value = f?.current ?? null;
        const last = f?.path?.[f.path.length - 1];
        extra = last ? `In 6 h <b>${num(last.predicted, t.dec)}</b> (${num(last.low, t.dec)}–${num(last.high, t.dec)})` : "";
      }
      const st = value === null ? "na" : t.status(value);
      const stTxt = { ok: "in normal range", warn: "outside normal range", bad: "past a danger line", na: "no value" }[st];
      return `<div class="card tile">
        <span class="ic"><svg viewBox="0 0 24 24" aria-hidden="true">${ICONS[t.icon]}</svg></span>
        <div>
          <div class="v">${value === null ? "–" : num(value, t.dec)} <small>${t.unit}</small><span class="dot ${st}" title="${stTxt}" aria-label="${stTxt}"></span></div>
          <div class="l">${t.label}</div>
          <div class="x">${extra}</div>
        </div></div>`;
    }).join("");
  }

  function renderGauges(r, thr) {
    const sbp = r.forecast.sbp?.current ?? null, map = r.forecast.map?.current ?? null;
    const h = r.history, i = h.hour.length - 1;
    let dbp = null;
    for (let k = i; k >= Math.max(0, i - 4); k--) if (h.dbp[k] !== null) { dbp = h.dbp[k]; break; }
    $("#bpTitle").textContent = `Blood pressure = ${num(sbp)}/${num(dbp)} mmHg${map !== null ? ` (MAP ${num(map)})` : ""}`;
    const t = state.cfg.thresholds;
    let st = ["ok", "Normal"];
    if ((sbp !== null && sbp < t.sbp_low) || (map !== null && map < t.map_low)) st = ["bad", "Low"];
    else if (sbp !== null && sbp >= 180) st = ["bad", "Very high"];
    else if (sbp !== null && sbp >= 140) st = ["warn", "High"];
    $("#bpStatus").innerHTML = `<span class="pill ${st[0]}">${st[1]}</span>`;
    const R = css("--red"), Y = css("--yellow"), G = css("--green");
    $("#gSys").innerHTML = gauge({ min: 40, max: 200, value: sbp, text: `${num(sbp)} mmHg`, ticks: [40, 80, 120, 160, 200],
      zones: [{ to: 90, color: R }, { to: 140, color: G }, { to: 180, color: Y }, { to: 200, color: R }] });
    $("#gDia").innerHTML = gauge({ min: 20, max: 130, value: dbp, text: `${num(dbp)} mmHg`, ticks: [20, 50, 75, 100, 130],
      zones: [{ to: 45, color: R }, { to: 90, color: G }, { to: 110, color: Y }, { to: 130, color: R }] });

    const risk = r.risk_probability, max = Math.max(40, Math.ceil((risk * 100) / 20) * 20);
    const lvl = { HIGH: ["bad", "High · alert"], MEDIUM: ["warn", "Medium"], LOW: ["ok", "Low"] }[r.risk_level];
    $("#riskStatus").innerHTML = `<span class="pill ${lvl[0]}">${lvl[1]}</span>`;
    $("#gRisk").innerHTML = gauge({ min: 0, max, value: risk * 100, text: pct(risk),
      ticks: [0, max / 4, max / 2, (3 * max) / 4, max], tickFmt: (v) => `${Math.round(v)}%`,
      zones: [{ to: (thr * 100) / 2, color: G }, { to: thr * 100, color: Y }, { to: max, color: R }] });
    $("#riskSub").innerHTML = `${r.relative_risk.toFixed(1)}× the average ICU hour · alert at ${pct(thr)}`;
  }

  function renderBanner(r) {
    const b = $("#banner"), parts = [];
    if (r.currently_unstable || r.on_vasopressors)
      parts.push("<b>Already unstable or on vasopressors.</b> The model is built to warn before deterioration, so read this risk with caution.");
    if (r.actual) {
      const a = r.actual, H = r.horizon_hours, ev = a.next_event_hour;
      if (ev && ev <= r.hour_now + H) parts.push(`<b>What happened next:</b> deterioration began at ICU hour ${ev}, ${ev - r.hour_now} h later.`);
      else if (a.unstable_hours || a.vasopressor_hours)
        parts.push(`<b>What happened next:</b> vital signs were past a danger line in ${a.unstable_hours} of the next ${H} hours${a.vasopressor_hours ? `, on vasopressors for ${a.vasopressor_hours}` : ""}. No new deterioration was counted because the patient was already unstable.`);
      else parts.push(`<b>What happened next:</b> vital signs stayed on the safe side of every danger line for ${H} hours.`);
    }
    b.hidden = !parts.length;
    b.className = `banner${r.currently_unstable || r.on_vasopressors ? "" : " info"}`;
    b.innerHTML = parts.join("<br>");
  }

  // ---------------------------------------------------------------- charts
  const whiteBg = { id: "whiteBg", beforeDraw(c) { const x = c.ctx; x.save(); x.fillStyle = "#fff"; x.fillRect(0, 0, c.width, c.height); x.restore(); } };
  const nowLine = (now) => ({ id: "now", afterDatasetsDraw(c) {
    const x = c.scales.x.getPixelForValue(now), { top, bottom } = c.chartArea, k = c.ctx;
    k.save(); k.strokeStyle = "#98A2B3"; k.lineWidth = 1; k.beginPath(); k.moveTo(x, top); k.lineTo(x, bottom); k.stroke();
    k.fillStyle = "#667085"; k.font = `11px ${css("--font")}`; k.fillText("now", x + 4, top + 11); k.restore(); } });
  const pointLabels = (dsIndexes, dec) => ({ id: "labels", afterDatasetsDraw(c) {
    const k = c.ctx; k.save(); k.font = `500 11.5px ${css("--font")}`; k.fillStyle = css("--ink-2"); k.textAlign = "center";
    for (const i of dsIndexes) {
      const meta = c.getDatasetMeta(i);
      meta.data.forEach((el, j) => { const v = c.data.datasets[i].data[j]; if (j === 0 && i === 2) return;
        if (v && v.y !== null) k.fillText(typeof dec === "function" ? dec(v.y) : Number(v.y).toFixed(dec), el.x, el.y - 9); });
    }
    k.restore(); } });

  function axes(xMin, xMax, extraY = {}) {
    return {
      x: { type: "linear", min: xMin, max: xMax, grid: { color: "#EEF0F3" }, ticks: { color: "#475467", stepSize: 2, font: { size: 12 } },
           title: { display: true, text: "ICU hour", color: "#667085", font: { size: 12 } } },
      y: { grid: { color: "#EEF0F3" }, ticks: { color: "#475467", font: { size: 12 }, maxTicksLimit: 7 }, ...extraY },
    };
  }

  function renderTabs() {
    $("#vitalTabs").innerHTML = VITALS.map((v) =>
      `<button role="tab" aria-selected="${v.key === state.vital}" data-v="${v.key}">${v.tab || v.label}</button>`).join("");
    $("#vitalTabs").querySelectorAll("button").forEach((b) => b.addEventListener("click", () => {
      state.vital = b.dataset.v; renderTabs(); renderLine(state.res);
    }));
  }

  function renderLine(r) {
    const v = VITALS.find((x) => x.key === state.vital), f = r.forecast[v.key];
    const color = css(v.color), now = r.hour_now, H = r.history;
    $("#lineTitle").textContent = `${v.label}: last 24 hours and next 6 hours`;
    $("#lineLegend").innerHTML = `<span><i class="sw solid" style="background:${color}"></i>Recorded</span>
      <span><i class="sw dash" style="border-color:${color}"></i>Forecast</span>
      <span><i class="sw band" style="background:${color}"></i>80% range</span>
      <span><i class="sw dangerl"></i>Danger line</span>${r.actual ? '<span><i class="sw actual"></i>What happened</span>' : ""}`;
    state.charts.line?.destroy();
    if (!f) return;
    const hist = H.hour.map((h, i) => ({ x: h, y: H[v.key][i] })).filter((p) => p.y !== null);
    const start = f.current ?? f.path[0].predicted;
    const xs = [now, ...f.path.map((p) => now + p.hours_ahead)];
    const lo = [start, ...f.path.map((p) => p.low)], hi = [start, ...f.path.map((p) => p.high)], md = [start, ...f.path.map((p) => p.predicted)];
    const [dir, key, fixed] = v.danger, dl = key ? state.cfg.thresholds[key] : fixed;
    const xMin = Math.max(0, now - 23), xMax = now + r.horizon_hours;
    const ds = [
      { data: xs.map((x, i) => ({ x, y: lo[i] })), borderWidth: 0, pointRadius: 0 },
      { data: xs.map((x, i) => ({ x, y: hi[i] })), borderWidth: 0, pointRadius: 0, fill: "-1", backgroundColor: color + "2E" },
      { data: xs.map((x, i) => ({ x, y: md[i] })), borderColor: color, borderDash: [6, 4], borderWidth: 2, pointRadius: 4,
        pointBackgroundColor: "#fff", pointBorderColor: color, pointBorderWidth: 2 },
      { data: hist, borderColor: color, borderWidth: 2.2, pointRadius: 3.5, pointBackgroundColor: "#fff", pointBorderColor: color, pointBorderWidth: 2 },
      { data: [{ x: xMin, y: dl }, { x: xMax, y: dl }], borderColor: css("--red"), borderDash: [2, 3], borderWidth: 1.5, pointRadius: 0 },
    ];
    if (r.actual) ds.push({ type: "scatter", data: r.actual.hour.map((h, i) => ({ x: h, y: r.actual[v.key][i] })).filter((p) => p.y !== null),
      pointRadius: 4, pointBackgroundColor: css("--ink"), borderColor: css("--ink") });
    const ys = [...hist.map((p) => p.y), ...lo, ...hi, dl, ...(r.actual ? r.actual[v.key].filter((y) => y !== null) : [])];
    const pad = (Math.max(...ys) - Math.min(...ys)) * 0.12 || 1;
    state.charts.line = new Chart($("#lineChart"), {
      type: "line", data: { datasets: ds },
      options: { animation: false, responsive: true, maintainAspectRatio: false,
        plugins: { legend: { display: false }, tooltip: { enabled: false } },
        scales: axes(xMin, xMax, { min: Math.floor(Math.min(...ys) - pad), max: Math.ceil(Math.max(...ys) + pad) }) },
      plugins: [whiteBg, nowLine(now), pointLabels([2], v.dec || 0)],
    });
  }

  function renderRiskBars(r, thr) {
    const rh = r.risk_history, n = rh.hour.length, from = Math.max(0, n - 24);
    const hours = rh.hour.slice(from), vals = rh.risk.slice(from).map((x) => x * 100);
    const col = vals.map((v) => v >= thr * 100 ? css("--red") : v >= (thr * 100) / 2 ? css("--orange") : css("--teal"));
    state.charts.risk?.destroy();
    const ymax = Math.max(thr * 100 * 1.6, ...vals) * 1.15;
    state.charts.risk = new Chart($("#riskChart"), {
      data: { labels: hours, datasets: [
        { type: "bar", data: vals, backgroundColor: col, barPercentage: 0.82, categoryPercentage: 1, order: 2 },
        { type: "line", data: hours.map(() => thr * 100), borderColor: css("--orange"), borderWidth: 2.5, pointRadius: 4.5,
          pointBackgroundColor: "#fff", pointBorderColor: css("--orange"), pointBorderWidth: 2, order: 1 },
      ] },
      options: { animation: false, responsive: true, maintainAspectRatio: false,
        plugins: { legend: { display: false },
                   tooltip: { callbacks: { title: (c) => `ICU hour ${c[0].label}`, label: (c) => `${c.parsed.y.toFixed(1)}%` } } },
        scales: {
          x: { type: "category", grid: { display: false }, ticks: { color: "#475467", font: { size: 12 }, autoSkip: true, maxRotation: 0 },
               title: { display: true, text: "ICU hour", color: "#667085", font: { size: 12 } } },
          y: { min: 0, max: Math.ceil(ymax / 5) * 5, grid: { color: "#EEF0F3" },
               ticks: { color: "#475467", font: { size: 12 }, callback: (v) => `${v}%` },
               title: { display: true, text: "Risk of deterioration in next 6 h", color: "#667085", font: { size: 12 } } },
        } },
      plugins: [whiteBg, barLabels],
    });
  }

  const barLabels = { id: "barLabels", afterDatasetsDraw(c) {
    const k = c.ctx, meta = c.getDatasetMeta(0), small = meta.data.length > 16;
    k.save(); k.font = `500 ${small ? 10.5 : 12}px ${css("--font")}`; k.fillStyle = css("--ink-2"); k.textAlign = "center";
    meta.data.forEach((el, j) => { const y = c.data.datasets[0].data[j]; k.fillText(y >= 10 ? y.toFixed(0) : y.toFixed(1), el.x, el.y - 6); });
    k.restore(); } };

  // ---------------------------------------------------------------- report
  function sections(md) {
    const out = {};
    md.split(/\n(?=### )/).forEach((chunk) => {
      const m = chunk.match(/^### (\d+)\.\s*(.*)\n([\s\S]*)$/);
      if (m) out[m[1]] = { title: m[2], body: m[3] };
    });
    return out;
  }

  function renderReport(r) {
    const s = sections(r.detailed_report);
    const lede = $("#lede");
    lede.className = `lede ${({ HIGH: "high", MEDIUM: "medium" })[r.risk_level] || ""}`;
    lede.textContent = r.summary.replace(/\s*Research prototype.*$/, "");
    const md = (ids) => ids.filter((i) => s[i]).map((i) => `### ${s[i].title}\n${s[i].body}`).join("\n\n");
    const html = (txt) => (window.marked ? marked.parse(txt) : `<pre>${esc(txt)}</pre>`);
    $("#reportLeft").innerHTML = html(md(["1", "2", "4"]));
    $("#reportRight").innerHTML = html(md(["5", "6"]));

    const hours = r.hourly_outlook;
    let t = `<thead><tr><th>Vital</th><th>Now</th>${hours.map((h) => `<th>+${h.hours_ahead} h<br>hour ${h.icu_hour}</th>`).join("")}</tr></thead><tbody>`;
    for (const v of VITALS) {
      const f = r.forecast[v.key];
      if (!f) continue;
      t += `<tr><td class="vname">${v.tab || v.label} <span>${v.unit}</span></td><td>${num(f.current, v.dec)}</td>`;
      for (const h of hours) {
        const p = h.vitals[v.key], w = h.warnings.find((x) => x.vital === v.key);
        let real = "";
        if (r.actual) {
          const i = r.actual.hour.indexOf(h.icu_hour), a = i >= 0 ? r.actual[v.key][i] : null;
          if (a !== null && a !== undefined) {
            const ok = a >= p.low && a <= p.high;
            real = `<span class="real${ok ? "" : " out"}">real ${num(a, v.dec)} ${ok ? "✓" : "✗"}</span>`;
          }
        }
        t += p ? `<td class="${w ? w.level : ""}">${num(p.predicted, v.dec)}<span>${num(p.low, v.dec)}–${num(p.high, v.dec)}</span>${real}</td>` : "<td>–</td>";
      }
      t += "</tr>";
    }
    $("#hourly").innerHTML = t + "</tbody>";
  }

  // ---------------------------------------------------------------- downloads
  const pdfUrl = () => `/api/patient/${state.stay}/report.pdf?hour=${state.hour}&reveal=${state.reveal}`;
  function saveBlob(blob, name) {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob); a.download = name;
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }
  function downloadCsv() {
    const r = state.res, keys = VITALS.map((v) => v.key);
    const head = ["icu_hour", "type", ...keys.flatMap((k) => [k, `${k}_low`, `${k}_high`]), "risk_pct_next_6h"];
    const riskAt = Object.fromEntries(r.risk_history.hour.map((h, i) => [h, r.risk_history.risk[i]]));
    const rows = [head];
    r.history.hour.forEach((h, i) => rows.push([h, "recorded", ...keys.flatMap((k) => [r.history[k][i] ?? "", "", ""]),
      riskAt[h] !== undefined ? (riskAt[h] * 100).toFixed(2) : ""]));
    r.hourly_outlook.forEach((h) => rows.push([h.icu_hour, "forecast",
      ...keys.flatMap((k) => { const p = h.vitals[k]; return p ? [p.predicted.toFixed(1), p.low.toFixed(1), p.high.toFixed(1)] : ["", "", ""]; }), ""]));
    const csv = rows.map((row) => row.map((c) => /[",\n]/.test(String(c)) ? `"${String(c).replace(/"/g, '""')}"` : c).join(",")).join("\n");
    saveBlob(new Blob([csv], { type: "text/csv" }), `icu-data_stay-${state.stay}_hour-${r.hour_now}.csv`);
  }
  function downloadPng(id) {
    const c = state.charts[id === "lineChart" ? "line" : "risk"];
    if (!c) return;
    const a = document.createElement("a");
    a.href = c.toBase64Image("image/png", 1);
    a.download = `${id === "lineChart" ? state.vital : "risk"}_stay-${state.stay}_hour-${state.res.hour_now}.png`;
    a.click();
  }

  // ---------------------------------------------------------------- data flow
  async function load() {
    const btn = $("#refreshBtn");
    btn.classList.remove("spin"); void btn.offsetWidth; btn.classList.add("spin");
    try {
      state.res = await api(`/api/patient/${state.stay}?hour=${state.hour}&reveal=${state.reveal}`);
      state.hour = state.res.hour_now;
      history.replaceState(null, "", `?stay=${state.stay}&hour=${state.hour}`);
      render();
    } catch (e) {
      $("#lede").textContent = `Couldn't load this patient: ${e.message}`;
    }
  }

  function setHour(h) {
    const p = state.patients.find((x) => x.stay_id === state.stay);
    state.hour = Math.max(3, Math.min(p ? p.max_hour : h, h));
    load();
  }

  async function init() {
    if (window.Chart) Chart.defaults.font.family = css("--font");
    renderTabs();
    try {
      [state.cfg, state.patients] = await Promise.all([api("/api/config"), api("/api/patients")]);
    } catch (e) {
      $("#refreshed").textContent = "Data unavailable";
      $("#lede").textContent = `Couldn't reach the model: ${e.message}. Run the pipeline, then restart the server.`;
      return;
    }
    const q = new URLSearchParams(location.search);
    const want = +q.get("stay");
    const p = state.patients.find((x) => x.stay_id === want) || state.patients[0];
    state.stay = p.stay_id;
    state.hour = q.get("hour") ? +q.get("hour") : p.default_hour;
    $("#patientSel").innerHTML = state.patients.map((x) =>
      `<option value="${x.stay_id}">${x.stay_id} · ${x.age} ${x.sex} · ${esc(x.unit.replace(/^.*\(([^)]+)\)$/, "$1"))}</option>`).join("");
    $("#patientSel").value = String(state.stay);

    $("#patientSel").addEventListener("change", (e) => {
      state.stay = +e.target.value;
      state.hour = state.patients.find((x) => x.stay_id === state.stay).default_hour;
      load();
    });
    $("#hourUp").addEventListener("click", () => setHour(state.hour + 1));
    $("#hourDown").addEventListener("click", () => setHour(state.hour - 1));
    $("#hourInput").addEventListener("change", (e) => setHour(+e.target.value || state.hour));
    $("#refreshBtn").addEventListener("click", load);
    $("#csvBtn").addEventListener("click", downloadCsv);
    $("#pdfBtn").addEventListener("click", () => {
      const b = $("#pdfBtn"); b.classList.add("busy"); b.lastChild.textContent = " Preparing PDF…";
      setTimeout(() => { b.classList.remove("busy"); b.lastChild.textContent = " Download PDF"; }, 3000);
    });
    document.querySelectorAll("[data-png]").forEach((b) => b.addEventListener("click", () => downloadPng(b.dataset.png)));

    const menu = $("#menu"), mb = $("#menuBtn");
    const closeMenu = () => { menu.hidden = true; mb.setAttribute("aria-expanded", "false"); };
    mb.addEventListener("click", (e) => { e.stopPropagation(); menu.hidden = !menu.hidden; mb.setAttribute("aria-expanded", String(!menu.hidden)); });
    document.addEventListener("click", (e) => { if (!menu.contains(e.target)) closeMenu(); });
    document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeMenu(); });
    $("#mPdf").addEventListener("click", (e) => { e.preventDefault(); closeMenu(); location.href = pdfUrl(); });
    $("#mCsv").addEventListener("click", (e) => { e.preventDefault(); closeMenu(); downloadCsv(); });
    $("#mPng").addEventListener("click", (e) => { e.preventDefault(); closeMenu(); downloadPng("lineChart"); setTimeout(() => downloadPng("riskChart"), 400); });
    $("#reveal").addEventListener("change", (e) => { state.reveal = e.target.checked; load(); });

    await load();
  }
  init();
})();
