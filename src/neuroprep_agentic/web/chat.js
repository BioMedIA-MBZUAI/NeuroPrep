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
    el("p", { class: "meta" }, run.moved_from
      ? `${run.moved_from} was already used, so results go to: ${run.run_root}`
      : run.default_output
        ? `No output folder named, so results go to: ${run.run_root}`
        : `Output folder: ${run.run_root}`),
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

// A question the assistant asked; the next message answers it.
let pending = null;
// The attached folder or files, already uploaded to the server.
let attachment = null;
const attachStatus = document.getElementById("attachment");
const DEFAULT_REQUEST = "Preprocess every modality in the attached data with quality control.";

function looksLikePath(text) {
  return /^[\/~]\S*$/.test(text);
}

// Fold an answer into the request the question was about.
function compose(answer) {
  if (!pending) return answer || (attachment ? DEFAULT_REQUEST : "");
  if (!answer) return pending.text;
  if (looksLikePath(answer)) {
    if (pending.bad_path) return pending.text.split(pending.bad_path).join(answer);
    const marker = pending.needs === "output_root" ? "Save outputs to" : "My data is at";
    return `${pending.text} ${marker} ${answer}.`;
  }
  return `${pending.text} ${answer}`;
}

function showAttachment() {
  attachStatus.replaceChildren();
  if (!attachment) return;
  const mb = (attachment.bytes / 1e6).toFixed(1);
  const label = attachment.done
    ? `Attached ${attachment.name}: ${attachment.count} file(s), ${mb} MB`
    : `Uploading ${attachment.name}: ${attachment.sent}/${attachment.count} file(s)…`;
  const remove = el("button", { type: "button", class: "text-button", "aria-label": "Remove attachment" }, "Remove");
  remove.addEventListener("click", () => { attachment = null; showAttachment(); });
  attachStatus.append(el("span", {}, label), attachment.done ? remove : "");
  if (attachment.error) attachStatus.append(el("span", { class: "error" }, ` ${attachment.error}`));
}

function randomId() {
  const bytes = crypto.getRandomValues(new Uint8Array(16));
  return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
}

async function upload(files) {
  const list = Array.from(files).filter((file) => !file.name.startsWith("."));
  if (!list.length) return;
  const first = list[0].webkitRelativePath || list[0].name;
  const name = list[0].webkitRelativePath ? first.split("/")[0] : list.length === 1 ? first : `${list.length} files`;
  const current = { id: randomId(), name, count: list.length, sent: 0, bytes: 0, done: false };
  attachment = current;
  send.disabled = true;
  showAttachment();
  try {
    for (const file of list) {
      const path = file.webkitRelativePath || file.name;
      const response = await fetch(`/api/upload?id=${current.id}&path=${encodeURIComponent(path)}`, {
        method: "POST",
        headers: { "Content-Type": "application/octet-stream" },
        body: file,
      });
      if (!response.ok) throw new Error((await response.json()).error || response.statusText);
      current.sent += 1;
      current.bytes += file.size;
      if (attachment === current) showAttachment();
    }
    current.done = true;
  } catch (error) {
    current.error = `Upload failed: ${error.message}`;
  } finally {
    send.disabled = false;
    if (attachment === current) showAttachment();
  }
}

for (const id of ["pick-folder", "pick-files"]) {
  const input = document.getElementById(id);
  input.addEventListener("change", () => { upload(input.files); input.value = ""; });
}
document.getElementById("attach-folder").addEventListener("click", () => document.getElementById("pick-folder").click());
document.getElementById("attach-files").addEventListener("click", () => document.getElementById("pick-files").click());

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const answer = prompt.value.trim();
  const usable = attachment && attachment.done ? attachment : null;
  if (!answer && !usable) return;
  const request = compose(answer);
  addMessage("user", answer || "", usable ? el("p", { class: "meta" }, `Attached: ${usable.name} (${usable.count} file(s))`) : "");
  send.disabled = true;
  try {
    const body = { request, execute: execute.value };
    if (usable) body.upload_id = usable.id;
    const response = await fetch("/api/auto", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || response.statusText);
    prompt.value = "";
    if (result.needs) {
      pending = { text: request, needs: result.needs, bad_path: result.bad_path };
      addMessage("bot", el("p", {}, result.question));
      prompt.placeholder = "Type your answer (a folder path is fine), or attach the data.";
      prompt.focus();
      return;
    }
    pending = null;
    attachment = null;
    showAttachment();
    prompt.placeholder = "What should NeuroPrep prepare? Include the input folder (and, optionally, where to save).";
    track(result);
  } catch (error) {
    addMessage("bot", el("p", { class: "error" }, `Something went wrong: ${error.message}. You can try again.`));
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
