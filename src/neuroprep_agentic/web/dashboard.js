"use strict";

const STAGES = [
  ["policy", "Understand request"],
  ["probe", "Inspect data"],
  ["pilot_selection", "Pick pilot"],
  ["pilot_plan", "Plan pilot"],
  ["pilot_run", "Run pilot"],
  ["evidence", "Sign evidence"],
  ["backend_selection", "Choose tools"],
  ["run_plan", "Plan full run"],
  ["run", "Process all"],
  ["release", "Release"],
];
const STATUS_LABEL = {
  complete: "done", running: "running", failed: "failed", blocked: "blocked",
  pending: "waiting", skipped: "skipped",
};
const RUN_MS = 3000;
const METRICS_MS = 5000;
const runId = new URLSearchParams(location.search).get("run") || "";
const $ = (id) => document.getElementById(id);
const tooltip = $("tooltip");

let run = null;
let phases = [];
let samples = [];
let selected = { phase: "run", task: "" };
let userPicked = false;
let lastFetch = 0;

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value !== undefined && value !== null) node.setAttribute(key, value);
  }
  for (const child of children) if (child !== null && child !== undefined && child !== "") node.append(child);
  return node;
}

function svg(tag, attrs = {}) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  return node;
}

function seconds(value) {
  const ms = Date.parse(value);
  return Number.isNaN(ms) ? null : ms / 1000;
}

function duration(total) {
  if (total === null || total === undefined || total < 0) return "–";
  const s = Math.round(total);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  if (h) return `${h}h ${String(m).padStart(2, "0")}m`;
  if (m) return `${m}m ${String(s % 60).padStart(2, "0")}s`;
  return `${s}s`;
}

function clock(t) {
  return new Date(t * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function shortSubject(id) {
  if (!id) return "dataset";
  const label = id.replace(/^sub-/, "");
  return `sub-${label.slice(0, 8)}`;
}

function nowSeconds() {
  return Date.now() / 1000;
}

function runStart() {
  const started = run && run.report && seconds(run.report.started_at);
  if (started) return started;
  const first = phases.flatMap((p) => p.tasks).map((t) => seconds(t.started_at)).filter(Boolean);
  return first.length ? Math.min(...first) : nowSeconds();
}

function runEnd() {
  if (run && run.state !== "running") {
    const done = run.report && seconds(run.report.finished_at);
    const stageTimes = Object.values((run.report || {}).stages || {}).map((s) => seconds(s.finished_at)).filter(Boolean);
    const taskTimes = phases.flatMap((p) => p.tasks).map((t) => seconds(t.finished_at)).filter(Boolean);
    const all = [done, ...stageTimes, ...taskTimes].filter(Boolean);
    if (all.length) return Math.max(...all);
  }
  return nowSeconds();
}

/* ---------- header, stepper, tiles ---------- */

function renderHead() {
  $("run-id").textContent = runId;
  $("run-title").textContent = run.request;
  $("run-paths").textContent = `Data: ${run.source_root}  →  Results: ${run.run_root}`;
  const state = $("run-state");
  const outcome = run.report && run.report.outcome;
  state.className = `chip ${run.state}`;
  state.textContent = outcome && run.state !== "running" ? `${run.state} · ${outcome}` : run.state;
  $("live-text").textContent = run.state === "running" ? "Live" : "Run ended";
  document.title = `${run.state === "running" ? "▶" : run.state === "failed" ? "✗" : "✓"} NeuroPrep Run Dashboard`;
}

// The report records a stage when it ends; a stage with no entry is done when a later
// stage has one, or (pilot/run planning) when that phase's tasks exist.
function stageStatus(stages, index) {
  const [key] = STAGES[index];
  if (stages[key]) return stages[key].status || "passed";
  const later = STAGES.slice(index + 1).some(([k]) => stages[k]);
  const hasPhase = (name) => phases.some((p) => p.phase === name && p.tasks.length);
  if (later || (key === "pilot_plan" && hasPhase("pilot")) || (key === "run_plan" && hasPhase("production"))) return "passed";
  return "pending";
}

function renderStepper() {
  const stages = (run.report || {}).stages || {};
  const list = $("stepper");
  list.replaceChildren();
  let current = run.state === "running";
  for (const [index, [key, label]] of STAGES.entries()) {
    const info = stages[key];
    let status = stageStatus(stages, index);
    const item = el("li", { title: info && info.error ? info.error : label }, label);
    if (status === "pending" && current) {
      status = "current";
      current = false;
    }
    if (info && info.finished_at) item.append(el("span", { class: "when" }, clock(seconds(info.finished_at))));
    else if (status === "current") item.append(el("span", { class: "when" }));
    item.className = status;
    list.append(item);
  }
}

function tile(label, value, sub) {
  return el("div", { class: "tile" }, el("div", { class: "label" }, label), el("div", { class: "value" }, value), sub ? el("div", { class: "sub" }, sub) : "");
}

function renderTiles() {
  const tasks = phases.flatMap((p) => p.tasks);
  const count = (status) => tasks.filter((t) => t.status === status).length;
  const gpu = samples.length && samples[samples.length - 1].gpus && samples[samples.length - 1].gpus[0];
  const failed = count("failed") + count("blocked");
  $("tiles").replaceChildren(
    tile("Tasks done", `${count("complete")}/${tasks.length}`, tasks.length ? `${Math.round((100 * count("complete")) / tasks.length)}%` : ""),
    tile("Running now", String(count("running")), phases.map((p) => `${p.phase}: ${p.tasks.filter((t) => t.status === "running").length}`).join(" · ")),
    tile("Failed / blocked", String(failed), failed ? "click a red row for its log" : "none"),
    tile("Waiting", String(count("pending")), ""),
    tile("GPU now", gpu && gpu.util_pct !== null ? `${gpu.util_pct}%` : "–", gpu ? `${gpu.mem_used_gb}/${gpu.mem_total_gb} GB · ${gpu.temp_c ?? "–"}°C` : "no GPU on this host"),
  );
}

function tickElapsed() {
  if (!run) return;
  $("run-elapsed").textContent = duration(runEnd() - runStart());
}

/* ---------- system charts ---------- */

const CHARTS = [
  { key: "gpu_util", title: "GPU utilization", unit: "%", max: () => 100, value: (s) => s.gpus && s.gpus[0] ? s.gpus[0].util_pct : null },
  { key: "gpu_mem", title: "GPU memory used", unit: "GB", max: (s) => (s.gpus && s.gpus[0] ? s.gpus[0].mem_total_gb : null), value: (s) => s.gpus && s.gpus[0] ? s.gpus[0].mem_used_gb : null },
  { key: "cpu", title: "CPU load", unit: "% of cores", max: () => 100, value: (s) => s.cpu_load_pct ?? null, soft: true },
  { key: "ram", title: "Memory used", unit: "GB", max: (s) => s.mem_total_gb ?? null, value: (s) => s.mem_used_gb ?? null },
];

function buildCharts() {
  const box = $("charts");
  for (const chart of CHARTS) {
    const card = el("div", { class: "chart", id: `chart-${chart.key}` },
      el("h3", {}, chart.title),
      el("div", { class: "now" }, "–"));
    const plot = svg("svg", { viewBox: "0 0 300 110", preserveAspectRatio: "none", role: "img", "aria-label": chart.title });
    card.append(plot);
    box.append(card);
    plot.addEventListener("mousemove", (event) => hoverChart(event, chart, plot));
    plot.addEventListener("mouseleave", () => { hideTip(); drawChart(chart); });
  }
}

function chartPoints(chart) {
  return samples.map((s) => ({ t: s.t, v: chart.value(s) })).filter((p) => p.v !== null && p.v !== undefined);
}

function chartScale(chart, points) {
  const last = samples[samples.length - 1] || {};
  let top = chart.max(last) || Math.max(1, ...points.map((p) => p.v));
  if (chart.soft) top = Math.max(top, ...points.map((p) => p.v));
  const t0 = points.length ? points[0].t : 0;
  const t1 = points.length ? points[points.length - 1].t : 1;
  const W = 300, H = 96, PAD = 2;
  return {
    top, t0, t1,
    x: (t) => (t1 === t0 ? W : ((t - t0) / (t1 - t0)) * W),
    y: (v) => H - PAD - (v / top) * (H - 2 * PAD),
  };
}

function drawChart(chart, hoverPoint) {
  const card = $(`chart-${chart.key}`);
  const plot = card.querySelector("svg");
  const points = chartPoints(chart);
  plot.replaceChildren();
  const now = card.querySelector(".now");
  if (!points.length) {
    now.textContent = "–";
    const na = svg("text", { x: 150, y: 55, "text-anchor": "middle", class: "na" });
    na.textContent = "not available on this host";
    plot.append(na);
    return;
  }
  const sc = chartScale(chart, points);
  const last = points[points.length - 1];
  now.textContent = `${Number(last.v).toFixed(chart.unit === "GB" ? 1 : 0)} ${chart.unit}`;
  for (const frac of [0, 0.5, 1]) {
    const yy = sc.y(sc.top * frac);
    plot.append(svg("line", { x1: 0, x2: 300, y1: yy, y2: yy, class: "gridline" }));
    const label = svg("text", { x: 298, y: yy - 2, "text-anchor": "end", class: "axis-label" });
    label.textContent = `${Math.round(sc.top * frac)}`;
    plot.append(label);
  }
  const d = points.map((p, i) => `${i ? "L" : "M"}${sc.x(p.t).toFixed(1)},${sc.y(p.v).toFixed(1)}`).join("");
  plot.append(svg("path", { d: `${d}L${sc.x(last.t).toFixed(1)},96L${sc.x(points[0].t).toFixed(1)},96Z`, class: "area" }));
  plot.append(svg("path", { d, class: "line", "vector-effect": "non-scaling-stroke" }));
  const start = svg("text", { x: 0, y: 108, class: "axis-label" });
  start.textContent = clock(points[0].t);
  const end = svg("text", { x: 300, y: 108, "text-anchor": "end", class: "axis-label" });
  end.textContent = clock(last.t);
  plot.append(start, end);
  if (hoverPoint) {
    plot.append(svg("line", { x1: sc.x(hoverPoint.t), x2: sc.x(hoverPoint.t), y1: 0, y2: 96, class: "cross", "vector-effect": "non-scaling-stroke" }));
    plot.append(svg("circle", { cx: sc.x(hoverPoint.t), cy: sc.y(hoverPoint.v), r: 4, class: "dot", "vector-effect": "non-scaling-stroke" }));
  }
}

function hoverChart(event, chart, plot) {
  const points = chartPoints(chart);
  if (!points.length) return;
  const box = plot.getBoundingClientRect();
  const sc = chartScale(chart, points);
  const fx = ((event.clientX - box.left) / box.width) * 300;
  let best = points[0];
  for (const p of points) if (Math.abs(sc.x(p.t) - fx) < Math.abs(sc.x(best.t) - fx)) best = p;
  drawChart(chart, best);
  showTip(event, [el("b", {}, `${Number(best.v).toFixed(1)} ${chart.unit}`), el("div", { class: "muted" }, `${chart.title} · ${clock(best.t)}`)]);
}

/* ---------- task timeline ---------- */

function taskWindow(task) {
  const start = seconds(task.started_at);
  if (!start) return null;
  const end = seconds(task.finished_at) || (task.status === "running" ? nowSeconds() : seconds(task.updated_at) || start);
  return [start, Math.max(end, start)];
}

function renderLegend() {
  const items = [["complete", "--status-good"], ["running", "--series-1"], ["failed", "--status-critical"], ["blocked", "--status-serious"], ["pending", null]];
  $("status-legend").replaceChildren(...items.map(([status, color]) => {
    const swatch = el("i");
    if (color) swatch.style.background = `var(${color})`;
    else swatch.style.border = "1px dashed var(--status-idle)";
    return el("span", {}, swatch, STATUS_LABEL[status]);
  }));
}

function renderTimeline() {
  const box = $("timeline");
  const tasks = phases.flatMap((p) => p.tasks);
  $("timeline-empty").hidden = tasks.length > 0;
  box.replaceChildren();
  if (!tasks.length) return;
  const t0 = runStart();
  const t1 = Math.max(runEnd(), t0 + 60);
  const span = t1 - t0;
  const pct = (t) => `${Math.min(100, Math.max(0, ((t - t0) / span) * 100))}%`;

  for (const phase of phases) {
    box.append(el("div", { class: "phase" }, `${phase.phase} · ${phase.tasks.filter((t) => t.status === "complete").length}/${phase.tasks.length} done`));
    const axis = el("div", { class: "axis" });
    for (let i = 0; i <= 4; i += 1) {
      const mark = el("span", {}, `+${duration((span * i) / 4)}`);
      mark.style.left = `${i * 25}%`;
      axis.append(mark);
    }
    box.append(el("div", { class: "axis-spacer" }), axis);
    for (const task of phase.tasks) {
      const isSelected = selected.phase === phase.phase && selected.task === task.task_id;
      const window_ = taskWindow(task);
      const took = window_ ? duration(window_[1] - window_[0]) : "";
      const label = el("div", { class: `row-label${isSelected ? " selected" : ""}` },
        el("span", { class: "name" }, task.executor_id),
        el("span", { class: "who" }, shortSubject(task.subject_id)),
        task.candidate ? el("span", { class: "tag", title: "compared against alternatives in the pilot" }, "compared") : "",
        el("span", { class: `status-text ${task.status}` }, `${STATUS_LABEL[task.status] || task.status}${took ? ` · ${took}` : ""}`));
      const track = el("div", { class: `track${isSelected ? " selected" : ""}` });
      if (window_) {
        const bar = el("div", { class: `bar ${task.status}` });
        bar.style.left = pct(window_[0]);
        bar.style.width = `max(4px, calc(${pct(window_[1])} - ${pct(window_[0])}))`;
        track.append(bar);
      } else if (task.status === "pending" || task.status === "blocked") {
        const bar = el("div", { class: `bar ${task.status === "blocked" ? "blocked" : "pending"}` });
        bar.style.left = pct(nowSeconds());
        bar.style.width = "4px";
        track.append(bar);
      }
      for (const node of [label, track]) {
        node.addEventListener("click", () => { userPicked = true; selectLog(phase.phase, task.task_id, `${task.executor_id} · ${shortSubject(task.subject_id)}`); });
        node.addEventListener("mousemove", (event) => showTip(event, taskTip(task, window_)));
        node.addEventListener("mouseleave", hideTip);
      }
      box.append(label, track);
    }
  }
}

function taskTip(task, window_) {
  const lines = [
    el("b", {}, `${task.executor_id} · ${task.subject_id || "dataset"}`),
    el("div", {}, `Status: ${STATUS_LABEL[task.status] || task.status}${task.attempts > 1 ? ` (attempt ${task.attempts})` : ""}`),
    el("div", { class: "muted" }, `Stages: ${(task.stages || []).join(", ") || "–"}`),
  ];
  if (window_) lines.push(el("div", { class: "muted" }, `Started ${clock(window_[0])} · ${duration(window_[1] - window_[0])}`));
  if (task.depends_on.length && task.status === "pending") lines.push(el("div", { class: "muted" }, `Waiting on ${task.depends_on.length} earlier task(s)`));
  if (task.error) lines.push(el("div", {}, `Error: ${task.error.slice(0, 240)}`));
  lines.push(el("div", { class: "muted" }, "Click for the live log"));
  return lines;
}

/* ---------- logs ---------- */

function selectLog(phase, task, title) {
  selected = { phase, task };
  $("log-title").textContent = title || "Run log";
  $("log").textContent = "Loading…";
  $("log-follow").checked = true;
  renderTimeline();
  refreshLog();
}

async function refreshLog() {
  const url = `/api/runs/${runId}/log?phase=${encodeURIComponent(selected.phase)}&task=${encodeURIComponent(selected.task)}`;
  try {
    const response = await fetch(url);
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || response.statusText);
    const log = $("log");
    const text = data.log || "(no output yet)";
    if (log.textContent !== text) {
      log.textContent = text;
      if ($("log-follow").checked) log.scrollTop = log.scrollHeight;
    }
    $("log-path").textContent = data.path ? data.path : selected.phase === "run" ? run && run.log_path : "";
  } catch (error) {
    $("log").textContent = `Could not read the log: ${error.message}`;
  }
}

/* ---------- tool choices and outputs ---------- */

function renderChoices() {
  const report = run.report || {};
  const tests = ((report.stages || {}).evidence || {}).tests || {};
  const decisions = ((report.stages || {}).backend_selection || {}).decisions || {};
  const rows = [];
  for (const [testId, result] of Object.entries(tests)) {
    const [, capability, backend] = testId.split("--");
    rows.push({ capability, backend, seconds: result.runtime_seconds, status: result.status, chosen: (decisions[capability] || {}).selected === backend, source: "signed" });
  }
  if (!rows.length) {
    // Before evidence is signed: live runtimes of the pilot's tasks.
    const pilot = phases.find((p) => p.phase === "pilot");
    for (const task of pilot ? pilot.tasks : []) {
      if (!["complete", "running", "failed"].includes(task.status) || task.executor_id.startsWith("neuroprep-bids") || task.executor_id === "bids-validator") continue;
      const window_ = taskWindow(task);
      rows.push({ capability: (task.stages || []).join(", "), backend: task.executor_id, seconds: window_ ? window_[1] - window_[0] : null, status: task.status, chosen: false, source: "live", candidate: task.candidate });
    }
  }
  const box = $("choices");
  if (!rows.length) return;
  rows.sort((a, b) => (a.capability || "").localeCompare(b.capability || "") || (a.seconds ?? 1e12) - (b.seconds ?? 1e12));
  const table = el("table", { class: "table" },
    el("thead", {}, el("tr", {}, el("th", {}, "Capability"), el("th", {}, "Tool"), el("th", {}, "Pilot time"), el("th", {}, "Result"), el("th", {}, ""))));
  const body = el("tbody");
  for (const row of rows) {
    body.append(el("tr", {},
      el("td", {}, row.capability || "–"),
      el("td", {}, row.backend),
      el("td", { class: "num" }, duration(row.seconds)),
      el("td", {}, el("span", { class: `chip ${row.status}` }, STATUS_LABEL[row.status] || row.status)),
      el("td", { class: "winner" }, row.chosen ? "✓ chosen" : "")));
  }
  table.append(body);
  box.replaceChildren(table, el("p", { class: "meta" }, rows[0].source === "signed" ? "From the signed pilot evidence." : "Live from the pilot; final once evidence is signed."));
}

function renderOutputs() {
  const block = $("outputs-block");
  block.hidden = !run.outputs.length;
  if (!run.outputs.length) return;
  $("outputs-count").textContent = `${run.outputs.length} file(s)`;
  const table = $("outputs");
  table.replaceChildren(el("thead", {}, el("tr", {}, el("th", {}, "File"), el("th", {}, "Tool"), el("th", {}, "Subject"))));
  const body = el("tbody");
  for (const row of run.outputs) body.append(el("tr", {}, el("td", { class: "path" }, row.path || ""), el("td", {}, row.executor_id || ""), el("td", {}, row.subject_id || "")));
  table.append(body);
}

/* ---------- tooltip ---------- */

function showTip(event, lines) {
  tooltip.replaceChildren(...lines);
  tooltip.hidden = false;
  const pad = 14;
  const width = tooltip.offsetWidth;
  const left = event.clientX + pad + width > window.innerWidth ? event.clientX - pad - width : event.clientX + pad;
  tooltip.style.left = `${Math.max(4, left)}px`;
  tooltip.style.top = `${event.clientY + pad}px`;
}

function hideTip() {
  tooltip.hidden = true;
}

/* ---------- polling ---------- */

async function pollRun() {
  try {
    const [status, tasks] = await Promise.all([
      fetch(`/api/runs/${runId}`).then((r) => (r.ok ? r.json() : Promise.reject(new Error("run not found on this server")))),
      fetch(`/api/runs/${runId}/tasks`).then((r) => r.json()),
    ]);
    run = status;
    phases = tasks.phases || [];
    lastFetch = Date.now();
    $("live-dot").hidden = false;
    followActiveTask();
    renderHead();
    renderStepper();
    renderTiles();
    renderTimeline();
    renderChoices();
    renderOutputs();
    tickElapsed();
    refreshLog();
  } catch (error) {
    $("run-title").textContent = runId ? `Cannot load run ${runId}: ${error.message}` : "No run selected: open this page from the chat.";
  }
  if (!run || run.state === "running") setTimeout(pollRun, RUN_MS);
}

async function pollMetrics() {
  try {
    const since = samples.length ? samples[samples.length - 1].t : 0;
    const data = await (await fetch(`/api/metrics?since=${since}`)).json();
    samples = samples.concat(data.samples || []).slice(-720);
    for (const chart of CHARTS) drawChart(chart);
    if (run) renderTiles();
  } catch (error) {
    /* keep the last samples; the next poll retries */
  }
  setTimeout(pollMetrics, METRICS_MS);
}

$("log-run").addEventListener("click", () => { userPicked = true; selectLog("run", "", "Run log"); });

// Until the viewer picks a log, follow the task that started most recently and is running.
function followActiveTask() {
  if (userPicked) return;
  let best = null;
  for (const phase of phases) {
    for (const task of phase.tasks) {
      if (task.status !== "running") continue;
      if (!best || (task.last_started_at || "") > (best.task.last_started_at || "")) best = { phase: phase.phase, task };
    }
  }
  if (best && (selected.phase !== best.phase || selected.task !== best.task.task_id)) {
    selectLog(best.phase, best.task.task_id, `${best.task.executor_id} · ${shortSubject(best.task.subject_id)} (live)`);
  }
}
setInterval(() => {
  tickElapsed();
  $("live-text").textContent = run && run.state === "running"
    ? (Date.now() - lastFetch > 15000 ? "Reconnecting…" : "Live")
    : $("live-text").textContent;
}, 1000);
buildCharts();
renderLegend();
pollRun();
pollMetrics();
