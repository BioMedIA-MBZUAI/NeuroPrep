const form = document.querySelector("#plan-form");
const submit = document.querySelector("#submit");
const errorBox = document.querySelector("#error");
const results = document.querySelector("#results");

document.querySelector("#example").addEventListener("click", () => {
  document.querySelector("#request").value = "Prepare every available structural, diffusion and resting-state acquisition across all visits. Prefer established pipelines, use a GPU when supported, and produce preprocessed images, QC reports and a model-ready index.";
  document.querySelector("#request").focus();
});

const setText = (selector, value) => { document.querySelector(selector).textContent = value; };
const renderPolicy = (policy) => {
  const list = document.querySelector("#policy");
  list.replaceChildren();
  const fields = {
    Objective: policy.objective,
    Outputs: (policy.requested_outputs || []).join(", "),
    "Preferred tools": (policy.preferred_packages || []).join(", ") || "Automatic",
    "GPU preference": policy.require_gpu_when_supported ? "Use when supported" : "Not required",
    "Experimental backends": policy.allow_experimental_backends ? "Allowed" : "Excluded",
  };
  Object.entries(fields).forEach(([key, value]) => {
    const term = document.createElement("dt"); term.textContent = key;
    const description = document.createElement("dd"); description.textContent = value;
    list.append(term, description);
  });
};

const renderBackends = (plan) => {
  const list = document.querySelector("#backends");
  list.replaceChildren();
  (plan.decisions || []).forEach((decision) => {
    const item = document.createElement("div"); item.className = "backend";
    const capability = document.createElement("strong"); capability.textContent = decision.capability;
    const selected = document.createElement("small");
    selected.textContent = decision.selected_package || "No compatible backend";
    item.append(capability, selected); list.append(item);
  });
  if (!list.children.length) list.textContent = "No backend decisions were required.";
};

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  errorBox.hidden = true;
  results.hidden = true;
  submit.disabled = true;
  submit.firstElementChild.textContent = "Compiling study…";
  const data = new FormData(form);
  const payload = {
    request: data.get("request"),
    source_root: data.get("source_root"),
    output_root: data.get("output_root"),
    dataset_id: data.get("dataset_id") || null,
    openjev_review: data.get("openjev_review") === "on",
  };
  try {
    const response = await fetch("/api/plan", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload),
    });
    const body = await response.json();
    if (!response.ok) throw new Error(body.error || "Planning failed");
    const summary = body.result;
    setText("#record-count", summary.record_count);
    setText("#task-count", summary.task_count);
    setText("#quarantine-count", summary.quarantined_count);
    setText("#unsupported-count", summary.unsupported_count);
    const badge = document.querySelector("#readiness");
    badge.textContent = summary.ready ? "Preflight ready" : "Review required";
    badge.classList.toggle("held", !summary.ready);
    renderPolicy(body.policy);
    renderBackends(body.backend_plan);
    setText("#preflight", JSON.stringify(body.preflight, null, 2));
    setText("#artifacts", JSON.stringify({
      execution_plan: summary.execution_plan,
      preflight: summary.preflight,
      state_path: summary.state_path,
      output_root: summary.binding.output_root,
    }, null, 2));
    results.hidden = false;
    results.scrollIntoView({behavior: "smooth", block: "start"});
  } catch (error) {
    errorBox.textContent = error.message;
    errorBox.hidden = false;
  } finally {
    submit.disabled = false;
    submit.firstElementChild.textContent = "Compile and validate plan";
  }
});
