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
