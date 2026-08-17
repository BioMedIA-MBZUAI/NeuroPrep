# Guarded backend qualification evidence

Backend qualification is target-bound. A completed representative run may
qualify a backend only when its catalog, executor command, resource contract,
container image, terminal task state, declared artifacts, and release indexes
agree with the target plan. A flat `{"test-id": "passed"}` file remains usable
for planning, but it is deliberately untrusted and cannot initialize or submit
a run.

## Evidence document

`build-backend-evidence` writes one deterministic, HMAC-SHA256-signed JSON
document. It contains:

- SHA-256 identities for the pilot plan, state database, release summary,
  qualification plan, declared artifacts, and container images;
- a read-only SQLite audit proving every pilot task is complete with return code
  zero and the exact task JSON/fingerprint recorded by the plan;
- release readiness plus artifact/tensor index counts;
- source-format and modality/protocol signatures for the pilot and target;
- the complete signed target smoke-test set and per-test pass status for the
  pilot/target executor intersection;
- the resolved exact target repository path and its current clean commit; and
- pilot git provenance only when a commit is embedded in the hashed pilot plan,
  state metadata, or release summary. If those immutable artifacts contain no
  commit, the document records `verification: unknown`, `commit: null`, and
  makes no pilot-commit claim.

The validator recomputes the signature, catalog digest, resolved target git
root, target git commit, target protocol, complete smoke-test set, normalized
executor commands and resources, image paths/hashes, and evidence digest. The compact verified
binding is serialized into both `backend-plan.json` and
`execution-plan.json`. The execution plan also retains the complete original
qualification set even after all tests pass.

## Key lifecycle

Keys are accepted only below `/nfs-stor/salma.hassan/` or
`/l/users/salma.hassan/`. They must be regular, non-symlink files with mode
`0600`; the key bytes are never serialized into plans, evidence, logs, or CLI
output.

Create a campaign key in an existing access-controlled directory:

```bash
neuroprep-agentic create-backend-evidence-key \
  --output /nfs-stor/salma.hassan/NeuroPrep-production/keys/backend-evidence-20260817.key
```

Rotate by creating a new path and `key-id`. Retain the old key while any plan
bound to it still needs standalone revalidation; remove it only after those
runs are initialized or retired. Never overwrite a key in place.

## Qualification and launch

First compile a planning-only target in a disposable probe root. Do not use
`--initialize` or `--submit` yet:

```bash
neuroprep-agentic prompt-run \
  --request 'My data is at /nfs-stor/salma.hassan/NeuroPrep-production/20260817/sources/anmerge; preprocess it and save to /nfs-stor/salma.hassan/NeuroPrep-production/20260817/runs/anmerge-full-v2-probe' \
  --manifest /nfs-stor/salma.hassan/NeuroPrep-production/manifests/anmerge-full-v2.json \
  --dataset-id anmerge-full-v2 \
  --agentic-project /nfs-stor/salma.hassan/NeuroPrep-agentic-production-COMMIT
```

Build the signed intersection from a terminal representative pilot:

```bash
neuroprep-agentic build-backend-evidence \
  --pilot-plan /nfs-stor/salma.hassan/NeuroPrep-agentic-benchmarks/20260815/openneuro-case01/execution-plan.json \
  --pilot-state /nfs-stor/salma.hassan/NeuroPrep-agentic-benchmarks/20260815/openneuro-case01/state/tasks.sqlite \
  --pilot-release /nfs-stor/salma.hassan/NeuroPrep-agentic-benchmarks/20260815/openneuro-case01/derivatives/release/release-summary.json \
  --target-plan /nfs-stor/salma.hassan/NeuroPrep-production/20260817/runs/anmerge-full-v2-probe/execution-plan.json \
  --git-repository /nfs-stor/salma.hassan/NeuroPrep-agentic-production-COMMIT \
  --signing-key /nfs-stor/salma.hassan/NeuroPrep-production/keys/backend-evidence-20260817.key \
  --key-id campaign-20260817 \
  --output /nfs-stor/salma.hassan/NeuroPrep-production/20260817/evidence/anmerge-full-v2.json
```

Recompile into a new, empty production root and initialize or submit with both
the document and its key:

```bash
neuroprep-agentic prompt-run \
  --request 'My data is at /nfs-stor/salma.hassan/NeuroPrep-production/20260817/sources/anmerge; preprocess it and save to /nfs-stor/salma.hassan/NeuroPrep-production/20260817/runs/anmerge-full-v2' \
  --manifest /nfs-stor/salma.hassan/NeuroPrep-production/manifests/anmerge-full-v2.json \
  --dataset-id anmerge-full-v2 \
  --agentic-project /nfs-stor/salma.hassan/NeuroPrep-agentic-production-COMMIT \
  --backend-evidence /nfs-stor/salma.hassan/NeuroPrep-production/20260817/evidence/anmerge-full-v2.json \
  --backend-evidence-key /nfs-stor/salma.hassan/NeuroPrep-production/keys/backend-evidence-20260817.key \
  --initialize
```

When initialization is invoked separately, `run-init` reopens and verifies the
signed source rather than trusting the embedded plan label:

```bash
neuroprep-agentic run-init /nfs-stor/salma.hassan/run/execution-plan.json \
  --backend-evidence-key /nfs-stor/salma.hassan/NeuroPrep-production/keys/backend-evidence-20260817.key
```

No evidence is required when the deterministic backend planner produces zero
target smoke tests. `--agentic-project` takes precedence over
`NEUROPREP_AGENTIC_PROJECT`, which in turn takes precedence over the historical
`/nfs-stor/salma.hassan/NeuroPrep-agentic` default. The selected path must
resolve to the exact top level of an existing git repository. Evidence built
for one path or commit cannot initialize a plan using another.
