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
  unique evaluation prompts, plus the exact 1,314-row prompt-token artifact
  requested once per arm;
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

For each arm, `collect.py accuracy` sends all 1,314 sealed GSM8K token-ID rows
once. It refuses caller-supplied dataset or token-artifact hashes that differ
from the contract, assigns a unique ordered request ID, preserves each raw
SGLang response and output-token IDs, and records the token-row hash and length.
The auditor hashes both sealed artifacts, matches every token row, checks arm
and echoed request identities, and independently re-scores every raw response
against the sealed answer labels instead of trusting the collector's booleans.
The KL
reference command separately generates 512 fixed output token IDs for each of
the 48 sealed helper prompts. It then teacher-forces those sequences on the
baseline; the candidate command teacher-forces the identical prompt and output
IDs. The rendered server command uses `kl_sink_server.py`, which creates a fresh
sealed per-arm sink root before SGLang starts. A qualification-only marker asks
the scheduler to write all ascending token IDs `0..vocab_size-1` directly from
the existing full-vocabulary input-logprob tensor. Each sample therefore needs
one teacher-forced forward, not one forward per transport chunk. The scheduler
writes fixed 8,192-ID, hash-addressed little-endian float32 shards and the HTTP
response carries only a small receipt. This is complete vocabulary coverage,
not selected-token evidence and not a top-k approximation.

The sink is disabled unless the launch adapter sets both halves of its sealed
authority. Its root must be an absolute, fresh child of an existing directory;
startup creates it and an immutable authority file exactly once. Requests carry
only a fixed in-vocabulary sample marker, never a path. Only TP rank zero writes,
and shard and sample-receipt files use exclusive creation, so retries, overwrite,
root reuse, mixed samples, or non-canonical position layouts fail closed.

The collector fails closed unless `/server_info` reports TP4 and `/model_info`
matches the exact model and tokenizer paths bound into the plan. The two
manifests must also share the sealed model-manifest hash, vocabulary size,
token-ID order, prompt hashes, output-token hashes, scored positions, and shard
boundaries. Every returned token ID must equal the requested ID, every log
probability must be finite, and each position's complete probability mass must
be within `5e-4` of one. The collector independently reads every server-written
shard, checks the small receipt and authority hashes, shape and complete token
range, then recomputes normalization before it writes the arm manifest.

The metric is the forward full-distribution divergence
`D_KL(P_baseline || Q_candidate)`. The auditor reads and hashes both sets of raw
shards, independently rechecks alignment and normalization, and computes a KL
value at every token position. It takes the arithmetic mean over positions in
each sealed sample, then applies the unchanged strict `< 0.0035` gate to the
maximum of the 48 sample means. The receipt and audit also report the mean of
the sample means and the maximum individual-position KL; neither replaces the
maximum-sample gate.

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

Create a private binding JSON with the staged model and tokenizer directories,
the tokenizer vocabulary size, the exact paths for every key in
`render_plan.ARTIFACT_HASH_KEYS`, separate baseline/candidate hosts, ports, and
rank-log paths, fresh absolute `kl_sink_roots.baseline` and
`kl_sink_roots.candidate` paths whose parents already exist, the candidate
FlashInfer Python path, CUDA version, GPU name, and container image. The
renderer launches both servers through the sealed sink adapter. Then, inside
the allocated compute job:

```bash
python -m tools.gdn_public_qualification.render_plan bindings.json --output plan.json
python -m tools.gdn_public_qualification.collect accuracy \
  --base-url "$BASELINE_URL" --arm baseline \
  --dataset "$GSM8K_DATASET" --dataset-sha256 "$GSM8K_DATASET_SHA256" \
  --prompt-ids "$GSM8K_PROMPT_IDS" \
  --prompt-ids-sha256 "$GSM8K_PROMPT_IDS_SHA256" \
  --output baseline-gsm8k.json
python -m tools.gdn_public_qualification.collect mtp-probe --help
python -m tools.gdn_public_qualification.collect kl-reference \
  --base-url "$BASELINE_URL" --input-ids "$LONG48_IDS" \
  --input-ids-sha256 "$LONG48_SHA256" --model-path "$MODEL_PATH" \
  --tokenizer-path "$TOKENIZER_PATH" --vocab-size "$VOCAB_SIZE" \
  --model-manifest-sha256 "$MODEL_MANIFEST_SHA256" \
  --sink-root "$BASELINE_SINK_ROOT" \
  --output "$BASELINE_SINK_ROOT/manifest.json"
python -m tools.gdn_public_qualification.collect kl-candidate \
  --base-url "$CANDIDATE_URL" \
  --reference "$BASELINE_SINK_ROOT/manifest.json" \
  --input-ids "$LONG48_IDS" --input-ids-sha256 "$LONG48_SHA256" \
  --model-path "$MODEL_PATH" --tokenizer-path "$TOKENIZER_PATH" \
  --vocab-size "$VOCAB_SIZE" \
  --model-manifest-sha256 "$MODEL_MANIFEST_SHA256" \
  --sink-root "$CANDIDATE_SINK_ROOT" \
  --output "$CANDIDATE_SINK_ROOT/manifest.json"
python -m tools.gdn_public_qualification.audit result.json --output audit.json
```

`result.json` references each KL manifest with a path relative to the result
file and its SHA256. Place the result so both sealed sink roots and their
manifests/shards are below its directory. The auditor resolves only safe
relative paths below the result directory and refuses missing, reused,
truncated, re-ordered, or hash-drifted distribution evidence.

Keep `bindings.json`, `plan.json`, raw model outputs, rank logs, results, and the
audit receipt in the durable campaign evidence directory. Record their SHA256
digests, scheduler job/step IDs, physical turnaround, and measured runtimes in
the final report. The final campaign remains embargoed until the pinned
FlashInfer commit has passed direct public validation on both required
architectures.
