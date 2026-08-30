# Public GDN qualification kit

This directory contains the source-owned client, immutable contract, and
independent receipt auditor for the final TP4 GDN A/B campaign. It invokes the
existing SGLang linear-attention backends; it does not alter dispatch or kernel
integration code.

The committed files contain only public object identities. Cluster paths live in
an uncommitted binding JSON. `render_plan.py` hashes every staged input before it
emits a private plan and pins:

- the exact SGLang integration base, FlashInfer commit/tree/bundle, exported
  source commit/tree, manifests, kernel, and exporter;
- the 1,319-row sealed GSM8K file as five fixed shots followed by exactly 1,314
  unique evaluation prompts, requested once per arm;
- the two fixed 32- and 48-prompt LongBench token-ID workloads;
- TP4 ranks 0–3 on compute capability 10.3 in the named container and model
  revision.

## Campaign shape

The control server uses the Triton prefill and decode backends. The candidate
uses the FlashInfer prefill and decode backends. Other server arguments are
identical. Do not add `--deterministic-inference`: SGLang deliberately rejects
that flag with the FlashInfer GDN prefill backend. Requests themselves use
temperature zero and sealed token IDs.

For each arm, `collect.py accuracy` runs all 1,314 GSM8K prompts once. The KL
client records baseline output-token log probabilities for all 48 sealed helper
prompts, then scores those exact token sequences on the candidate. It uses the
same non-negative approximation as SGLang's KL test utilities,
`exp(logr) - 1 - logr`, and the auditor recomputes its mean.

Each throughput workload follows `baseline, candidate, candidate, baseline`
four times, producing eight observations per arm per workload. An observation
serves all fixed prompts with 512 output tokens and reports output-token
throughput. The auditor pairs the two measurements within each ABBA block,
resamples the four paired block ratios within each workload with the fixed seed,
and performs exactly 20,000 bootstrap replicates. Both the aggregate geomean and
its lower 95% percentile bound must exceed one. A workload is a resolved
regression only when its upper 95% bootstrap bound is below one; no such workload
is allowed.

Candidate rank logs must report the same sorted, exact
`cake.gdn_prefill.noncp.*` and `cake.gdn_decode.noncp.*` route sets on ranks 0–3,
with no route error or fallback. Every baseline rank must report zero Cake
routes. Missing route evidence fails closed.

## Use

Create a private binding JSON with the staged model directory, the exact paths
for every key in `render_plan.ARTIFACT_HASH_KEYS`, separate baseline/candidate
ports and rank-log paths, the candidate FlashInfer Python path, CUDA version,
GPU name, and container image. Then, inside the allocated compute job:

```bash
python -m tools.gdn_public_qualification.render_plan bindings.json --output plan.json
python -m tools.gdn_public_qualification.collect --help
python -m tools.gdn_public_qualification.audit result.json --output audit.json
```

Keep `bindings.json`, `plan.json`, raw model outputs, rank logs, results, and the
audit receipt in the durable campaign evidence directory. Record their SHA256
digests, scheduler job/step IDs, physical turnaround, and measured runtimes in
the final report. The final campaign remains embargoed until the pinned
FlashInfer commit has passed direct public validation on both required
architectures.
