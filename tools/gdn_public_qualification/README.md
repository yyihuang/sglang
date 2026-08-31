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

The private binding names separate `server_hosts.baseline` and
`server_hosts.candidate` values. This represents the required two-node layout:
one resident TP4 server on each four-GPU GB300 node, so ABBA observations
alternate between already-loaded arms instead of including model-reload time.

## Campaign shape

The control server uses the Triton prefill, decode, and target-verify backends.
The candidate uses the FlashInfer prefill, decode, and target-verify backends.
Both servers run real NEXTN chain speculation with three draft steps, top-k 1,
and a four-token verify window; other server arguments are identical. Do not add
`--deterministic-inference`: SGLang deliberately rejects that flag with the
FlashInfer GDN prefill backend. Requests themselves use temperature zero and
sealed token IDs.

Before the long campaign, `collect.py mtp-probe` reads `/server_info` and fails
unless NEXTN has resolved to EAGLE with exactly the pinned TP4, draft, and
per-arm backend configuration. It then sends one eight-output-token, ignore-EOS
request from sealed LongBench prompt zero. The receipt preserves the output
token IDs, their hash, and wall runtime. Together with the exact per-rank T=4
route markers, this proves that the campaign exercised the real MTP/target-
verify server configuration rather than merely declaring it in a plan.

For each arm, `collect.py accuracy` runs all 1,314 GSM8K prompts once. The KL
client records baseline output-token log probabilities for all 48 sealed helper
prompts, then scores those exact token sequences on the candidate. The resulting
`exp(logr) - 1 - logr` quantity is SGLang's historical log-ratio surrogate, not
a full-vocabulary KL divergence. The receipt names it
`max_sample_sglang_logratio_surrogate`; the auditor recomputes every per-sample
mean and applies the unchanged exclusive `0.0035` gate to the maximum of the 48
samples. It also reports the campaign mean for continuity.

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
`flashinfer.gdn_prefill.noncp.*` and `flashinfer.gdn_decode.noncp.*` route sets on
ranks 0–3, including
`flashinfer.gdn_decode.noncp.indexed_bf16_verify_t4.tile16_fullwarp`, with no
route error or fallback. Every baseline rank must report zero attributable
optimized-route markers. Missing or malformed route evidence fails closed.

## Final route artifact

`produce_route_artifact.py` emits the deterministic
`gdn-noncp-final-sglang-route-artifact-v1` input consumed by the TP4 renderer.
It does not accept caller-written routes. It selects the four exact campaign
rows pinned in `contract.py` from a hash-authenticated final core manifest,
requires matching optimized SM100a/SM103a dispatch, and proves every selected
raw route against literals in the hash-authenticated installed
`flashinfer/jit/gdn_noncp.py`. Decode routes must be exact literals; prefill
routes use the loader's canonical base plus its generated `dvsplit` or
`full_dv` suffix. The exact T=4 decode literal is mandatory.

The command requires full Cake and FlashInfer commit/tree identities, clean
matching checkouts, the exact Cake exporter hash, core and overlay manifest
hashes, and an overlay whose complete output set matches the FlashInfer tree.
It rejects placeholders, abbreviations, dirty or mismatched sources, missing
contract rows, and stale literals. Output contains no timestamp or local path
and must be a fresh absolute file.

```bash
python -m tools.gdn_public_qualification.produce_route_artifact \
  --cake-root /absolute/clean/final-cake \
  --cake-commit "$CAKE_COMMIT" --cake-tree "$CAKE_TREE" \
  --cake-exporter-sha256 "$CAKE_EXPORTER_SHA256" \
  --flashinfer-root /absolute/clean/final-flashinfer \
  --flashinfer-commit "$FLASHINFER_COMMIT" --flashinfer-tree "$FLASHINFER_TREE" \
  --core-manifest /absolute/export/manifest.json \
  --core-manifest-sha256 "$CORE_MANIFEST_SHA256" \
  --overlay-manifest /absolute/export/overlay/manifest.json \
  --overlay-manifest-sha256 "$OVERLAY_MANIFEST_SHA256" \
  --output /absolute/fresh/route-artifact.json
```

Hash the result into the unresolved TP4 final pins. This producer does not
launch servers or alter route, accuracy, KL, or performance acceptance gates.

## Use

Create a private binding JSON with the staged model directory, the exact paths
for every key in `render_plan.ARTIFACT_HASH_KEYS`, separate baseline/candidate
hosts, ports, and rank-log paths, the candidate FlashInfer Python path, CUDA
version, GPU name, and container image. Then, inside the allocated compute job:

```bash
python -m tools.gdn_public_qualification.render_plan bindings.json --output plan.json
python -m tools.gdn_public_qualification.collect mtp-probe --help
python -m tools.gdn_public_qualification.collect --help
python -m tools.gdn_public_qualification.audit result.json --output audit.json
```

Keep `bindings.json`, `plan.json`, raw model outputs, rank logs, results, and the
audit receipt in the durable campaign evidence directory. Record their SHA256
digests, scheduler job/step IDs, physical turnaround, and measured runtimes in
the final report. The final campaign remains embargoed until the pinned
FlashInfer commit has passed direct public validation on both required
architectures.
