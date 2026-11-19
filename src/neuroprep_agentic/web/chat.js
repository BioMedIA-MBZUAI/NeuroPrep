"use strict";

const STAGES = [
  ["policy", "Understand the request"],
  ["probe", "Inspect the data and plan"],
  ["pilot_selection", "Pick pilot subjects"],
  ["pilot_plan", "Plan the pilot"],
  ["pilot_run", "Run the pilot"],
  ["evidence", "Sign tool evidence"],
  ["backend_selection", "Choose tools"],
  ["run_plan", "Plan the full run"],
  ["run", "Process every subject"],
  ["release", "Release outputs"],
];
const POLL_MS = 5000;
const messages = document.getElementById("messages");
const form = document.getElementById("composer");
const prompt = document.getElementById("prompt");
const execute = document.getElementById("execute");
const send = document.getElementById("send");
const bubbles = new Map();

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  for (const child of children) node.append(child);
  return node;
}

function scrollDown() {
  window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
}

function addMessage(kind, ...children) {
  const item = el("li", { class: `msg ${kind}` }, ...children);
  messages.append(item);
  scrollDown();
  return item;
}

function stageDetail(values) {
  if (values.error) return values.error;
  if (values.subjects) return `${values.subjects.length} subject(s)`;
  if (values.counts) return Object.entries(values.counts).map(([k, v]) => `${v} ${k}`).join(", ");
  if (values.decisions) {
    return Object.entries(values.decisions)
      .map(([capability, d]) => `${capability} → ${d.selected}${d.compared ? " (compared)" : ""}`)
      .join("; ");
  }
  if (values.reason) return values.reason;
  return "";
}

function renderRun(run) {
  const bubble = bubbles.get(run.run_id);
  bubble.replaceChildren();
  const report = run.report || {};
  const stages = report.stages || {};
  const outcome = report.outcome ? ` · ${report.outcome}` : "";
  bubble.append(
    el("p", {}, el("span", { class: `state ${run.state}` }, run.state), ` ${run.execute}${outcome}`),
    el("p", { class: "meta" }, `Output folder: ${run.run_root}`),
  );
  const list = el("ol", { class: "run-stages" });
  for (const [key, label] of STAGES) {
    const values = stages[key];
    const status = values ? values.status || "passed" : "pending";
    const detail = values ? stageDetail(values) : "";
    list.append(
      el("li", { class: status }, label, detail ? el("span", { class: "detail" }, ` — ${detail}`) : ""),
    );
  }
  bubble.append(list);
  if (run.state === "failed" && !report.failed_stage) {
    bubble.append(el("p", { class: "error" }, `The run stopped (exit code ${run.exit_code}); see the log below.`));
  }
  if (run.outputs.length) {
    const table = el("table", { class: "outputs" },
      el("thead", {}, el("tr", {}, el("th", {}, "Released file"), el("th", {}, "Tool"), el("th", {}, "Subject"))));
    const body = el("tbody");
    for (const row of run.outputs) {
      body.append(el("tr", {},
        el("td", { class: "path" }, row.path || ""),
        el("td", {}, row.executor_id || ""),
        el("td", {}, row.subject_id || "")));
    }
    table.append(body);
    bubble.append(el("p", {}, `${run.outputs.length} released file(s):`), table);
  } else if (run.state === "finished") {
    bubble.append(el("p", { class: "meta" }, "Finished. Nothing was released (plan-only or Slurm runs release later)."));
  }
  if (run.log_tail) {
    bubble.append(el("details", {}, el("summary", {}, "Log"), el("pre", {}, run.log_tail)));
  }
}

async function poll(runId) {
  try {
    const response = await fetch(`/api/runs/${runId}`);
    const run = await response.json();
    if (!response.ok) throw new Error(run.error || response.statusText);
    renderRun(run);
    if (run.state === "running") setTimeout(() => poll(runId), POLL_MS);
  } catch (error) {
    setTimeout(() => poll(runId), POLL_MS * 3);
  }
}

function track(run) {
  bubbles.set(run.run_id, addMessage("bot"));
  renderRun(run);
  if (run.state === "running") setTimeout(() => poll(run.run_id), POLL_MS);
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = prompt.value.trim();
  if (!text) return;
  addMessage("user", text);
  send.disabled = true;
  try {
    const response = await fetch("/api/auto", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ request: text, execute: execute.value }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || response.statusText);
    prompt.value = "";
    track(result);
  } catch (error) {
    addMessage("bot", el("p", { class: "error" }, `Could not start: ${error.message}`));
  } finally {
    send.disabled = false;
  }
});

prompt.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) form.requestSubmit();
});

document.getElementById("example").addEventListener("click", (event) => {
  prompt.value = event.currentTarget.textContent;
  prompt.focus();
});

fetch("/api/runs")
  .then((response) => response.json())
  .then((data) => {
    for (const run of data.runs || []) {
      addMessage("user", run.request);
      track(run);
    }
  })
  .catch(() => {});
