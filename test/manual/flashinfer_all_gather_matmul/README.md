# FlashInfer all-gather matmul Llama-70B experiment

This manual harness measures a TP4 Llama-3.1-70B-Instruct Q-projection route
on four SM100/SM103 GPUs. It is deliberately an experiment, not a production
SGLang dispatch path.

The three arms are:

- `native`: unchanged SGLang packed QKV projection; no artificial collective
  and no source patch is loaded.
- `explicit`: split the replicated hidden states into contiguous token shards,
  reconstruct them with `all_gather_into_tensor`, run the original local Q
  GEMM, run the original local KV GEMM, and concatenate Q and KV.
- `candidate`: use `flashinfer.comm.all_gather_matmul(..., backend="cake")`
  for the same Q result, then the same local KV GEMM and concatenation.

For global hidden states `X` and the TP-local packed weight `[Wq; Wkv]`, the
explicit and candidate arms both return
`concat(X @ Wq.T, X @ Wkv.T)`. The first routed call through each of the 80
layers is checked against the unchanged packed `linear(X, [Wq; Wkv])` with
BF16 tolerances `atol=rtol=1e-2`.

Only candidate versus explicit is the fused-kernel speedup. Candidate versus
native is reported separately as the whole-server cost of adding this
experimental sequence-parallel boundary. A production route would require
hidden states that are already scattered, so this harness does not by itself
justify changing SGLang's default Llama path.

## Fixed inputs

- Model: `meta-llama/Llama-3.1-70B-Instruct`
- Revision: `1605565b47bb9346c5515c34102e054115b4f98b`
- 30-shard LFS manifest SHA256:
  `438a360fe1b9c1f748fdd543757f11eb677c453cc522aa61100c4a5e6dce2c6f`
- Model config SHA256:
  `fa6e9124e4621df77aecf96fbfaf7975814013d2d5ab1c972e965000588a9749`
- Safetensors index SHA256:
  `2abe0910e23770a30ccf9b1b91804c64831c47f9c98defaa5293aa999433fc2b`
- The four tokenizer/generation metadata files are also hashed at use; their
  exact digests are recorded in `input-contract.json`.
- GSM8K SHA256: `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`
- ShareGPT SHA256: `35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4`
- TP: 4, BF16, seed `20260825`
- Accuracy: first 500 examples of a pinned GSM8K test file, 5-shot,
  temperature 0, native `/generate`, maximum 512 new tokens
- Parity probe: the first 32 of those exact prompts, 32 new tokens plus token
  ids and output log probabilities
- Serving benchmark: pinned ShareGPT source, 256 requests, fixed 4096 input
  and 128 output tokens, concurrency 64, request rate infinity, one warmup and
  three measured repetitions

The run must additionally pass all four per-rank route counters and find
`kernel_cake_blackwell_all_gather_matmul_bfloat16_ws4` in a GPU activity trace.
The parity gate requires 32/32 exact token-id sequences and a maximum aligned
output-logprob delta of 0.05. `input_contract.py` verifies both dataset hashes,
the 30-shard model LFS manifest, and records the exact 256-request serving
selection hash. `run_all_variants.sh` is intended to run only inside a Slurm
GPU allocation with real weights.

## Fail-closed execution contract

`run_all_variants.sbatch` requires a digest-pinned container, an exact clean
SGLang commit and tree, and a candidate wheel receipt containing exact
FlashInfer commit, tree, wheel SHA256, and public API signature. The wheel is
force-installed without dependencies into a fresh node-local Python target;
the runtime contract rejects imports from any other installation. It also
requires exactly four unique GB200/B200 UUIDs with compute capability 10.0.

The pinned image provides a locally versioned `flashinfer-cubin` whose PEP 440
base version is `0.6.14`, while the isolated candidate provides
`flashinfer-python==0.6.18` and bundles the Cake JIT source.
The job therefore sets `FLASHINFER_DISABLE_VERSION_CHECK=1` only for this exact
version split. The runtime receipt records both distributions and the bypass,
then still requires the public API to come from the isolated wheel, the exact
Cake ws4 backend and kernel sources to be present there, and the SM100 source to
declare `SMEM_TOTAL` as 197632 bytes.

The result root, each arm, and each profile directory must not exist before the
run. Every arm requires a bounded, clean process-group shutdown and writes
`COMPLETE` only after its server shuts down and all gates pass; the top-level
marker is written only after three-arm aggregation passes. Completion receipts
use exclusive/no-follow creation. The top-level receipt binds the summary and
inventory hashes, runtime and input receipt hashes, source identity, container
digest, and Slurm job.
The aggregator requires exactly one warmup result with 32 completed requests,
exactly three measured results with 256 completed requests, finite positive
metrics, no request errors, 500 GSM8K examples, a minimum GSM8K score of 0.90,
and a maximum cross-arm score delta of 0.002. It hashes all raw artifacts.

The candidate CUPTI evidence must consist of exactly one GPU activity trace
for each TP rank 0 through 3, with the Cake ws4 symbol present on every rank.
This trace is route evidence only; performance is derived exclusively from the
three measured end-to-end serving repetitions.

The final summary keeps timing scopes separate: scheduler queue time, physical
turnaround from submission through summary completion, allocation physical
time, the sum of the three arm wall times, and the measured serving runtime.
