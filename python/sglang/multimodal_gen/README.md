<div align="center"  style="display:block; margin:auto;">
<img src=https://github.com/lm-sys/lm-sys.github.io/releases/download/test/sgl-diffusion-logo.png width="80%"/>
</div>

**SGLang diffusion is an inference framework for accelerated image/video generation.**

SGLang diffusion features an end-to-end unified pipeline for accelerating diffusion models. It is designed to be modular and extensible, allowing users to easily add new models and optimizations.

## Key Features

SGLang Diffusion has the following features:
  - Broad model support: Wan, FastWan, FLUX, Qwen-Image, Z-Image, Ideogram 4, Krea-2, Cosmos3, LTX-2/LTX-2.3, MiniMax-H3, LingBot Video MoE, LingBot World, SANA-Video/SANA-WM, JoyEcho, MOVA, GLM-Image, ERNIE-Image, Hunyuan3D, and more
  - Fast inference speed: empowered by optimized `sgl-kernel` kernels, scheduler/runtime improvements, caching acceleration, and native diffusion hot-path optimizations
  - Ease of use: OpenAI-compatible api, CLI, and python sdk support
  - Multi-platform support:
    - NVIDIA GPUs (H100, H200, A100, B200, 4090, 5090)
    - AMD GPUs (MI300X, MI325X, MI355X)
    - Intel XPUs
    - Ascend NPU (A2, A3)
    - Apple Silicon (M-series via MPS)
    - Moore Threads GPUs (MTT S5000)

### AMD/ROCm Support

SGLang Diffusion supports AMD Instinct GPUs through ROCm. On AMD platforms, we use the Triton attention backend and leverage AITER kernels for optimized layernorm and other operations. See the [installation guide](https://docs.sglang.io/docs/sglang-diffusion/installation) for setup instructions.

### Moore Threads/MUSA Support

SGLang Diffusion supports Moore Threads GPUs (MTGPU) through the MUSA software stack. On MUSA platforms, we use FlashAttention (FA3) when available; also supports Sage Attention when installed; otherwise falls back to the Torch SDPA backend. See the [installation guide](https://docs.sglang.io/docs/sglang-diffusion/installation) for setup instructions.

### Apple MPS Support

SGLang Diffusion supports Apple Silicon (M-series) via the MPS backend. Since Triton is Linux-only, all Triton kernels are replaced with PyTorch-native fallbacks on MPS. Norm operations can be optionally accelerated with MLX fused Metal kernels (`SGLANG_USE_MLX=1`). See the [installation guide](https://docs.sglang.io/docs/sglang-diffusion/installation) for setup instructions.

## Getting Started

```bash
uv pip install 'sglang[diffusion]' --prerelease=allow
```

For more installation methods (e.g. pypi, uv, docker, ROCm/AMD, MUSA/Moore Threads), check the [installation guide](https://docs.sglang.io/docs/sglang-diffusion/installation).

## Inference

Here's a minimal example to generate a video using the default settings:

```python
from sglang.multimodal_gen import DiffGenerator

def main():
    # Create a diff generator from a pre-trained model
    generator = DiffGenerator.from_pretrained(
        model_path="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        num_gpus=1,  # Adjust based on your hardware
    )

    # Generate the video
    video = generator.generate(
        sampling_params_kwargs=dict(
            prompt="A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes wide with interest.",
            return_frames=True,  # Also return frames from this call (defaults to False)
            output_path="my_videos/",  # Controls where videos are saved
            save_output=True
        )
    )

if __name__ == '__main__':
    main()
```

Or, more simply, with the CLI:

```bash
sglang generate --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --component-residency text_encoder=component-offload --pin-cpu-memory \
    --prompt "A curious raccoon" \
    --save-output
```

### Wan hybrid attention for Wan

On B200/GB200 (`sm_100`) and B300/GB300 (`sm_103`), a FlashInfer build that
exports the public `flashinfer.wan_hybrid_attention` API can run the exact Wan
self-attention shape through the explicit hybrid backend:

```bash
sglang generate \
  --model-path nvidia/Wan2.2-T2V-A14B-Diffusers-NVFP4 \
  --attention-backend wan_hybrid \
  --prompt "A curious raccoon walks through a sunlit forest" \
  --save-output
```

The backend is intentionally fail-closed: it accepts caller-owned contiguous
BF16 NHD Q/K/V and output at exactly `B=1, S=4800, H=40, D=128`, with
noncausal dense self-attention and the default `1 / sqrt(128)` score scale.
Q/K remain BF16; FlashInfer owns the reusable FP4 V/P workspace and writes
directly into the caller's BF16 output. Wan cross-attention continues to use
the normal dense backend because its query and KV sequence lengths differ.
Packed-varlen, masks, GQA/MQA, and ring attention are not supported.

This integration remains explicit opt-in and the production route stays on FA.
Complete all-step/all-pair diffusion trajectories and generated frames must be
qualified against that production route; isolated attention accuracy is not a
model-level correctness claim. The
`wan_hybrid_min_timestep` and `wan_hybrid_layer_indices` backend options are
diagnostic gates. A run is not a valid hybrid qualification unless its reported
`wan_hybrid_hit_count` is greater than zero.

Use `compare_diffusion_trajectory_similarity` for model-level qualification.
The tool requires at least two warmup runs and five measured runs. Correctness
captures every trajectory step and evaluates every same-variant and
cross-variant run pair. Performance disables trajectory capture, executes both
reference-first and candidate-first orders, and passes only when both median
speedups are at least 1.0 and every measured candidate run reports a positive
backend hit count.

The qualification runner builds the fixed single-block, full-transformer, and
generation matrices without depending on a particular cluster layout. Pass the
staged public revisions explicitly so the resulting manifest records exactly
what was measured:

```bash
python -m sglang.multimodal_gen.tools.run_wan_hybrid_qualification \
  --model-path /models/Wan2.2-T2V-A14B-Diffusers-NVFP4 \
  --model-id nvidia/Wan2.2-T2V-A14B-Diffusers-NVFP4 \
  --output-dir /results/wan-hybrid \
  --sglang-revision "$SGLANG_REVISION" \
  --flashinfer-revision "$FLASHINFER_REVISION" \
  --staging-label "$STAGING_LABEL" \
  --scenario generation \
  --mode correctness
```

A correctness invocation always runs both execution orders as separate
generation-trajectory reports. A performance invocation runs one comparison
command with `--run-order both`; that command disables trajectory capture and
contains both orders in the same report. Repeat `--scenario` or `--mode` to
request more than one matrix entry. `single-block` selects block zero through
`wan_hybrid_layer_indices`, `full-transformer` selects the primary Wan
transformer component, and `generation` enables the backend for every eligible
Wan self-attention layer.

The runner's `full-transformer` scenario is still an end-to-end generation
trajectory with the primary transformer component routed to `wan_hybrid`. It is
not an independent transformer single-forward result. For that separate check,
use
`run_wan_transformer_forward_qualification`. The caller supplies already-loaded
reference and candidate models, the resolved component directory, and the
actual fixed mapping of model-forward keyword arguments. The harness invokes
both models directly with that same mapping; caller-owned forward closures are
not accepted. Run both
`transformer` and `transformer_2` in both explicit execution orders, write each
report with `write_wan_transformer_forward_report`, and pass all four paths to
a `full-transformer` runner invocation:

```bash
  --full-transformer-forward-report /results/transformer-reference-first.json \
  --full-transformer-forward-report /results/transformer-candidate-first.json \
  --full-transformer-forward-report /results/transformer-2-reference-first.json \
  --full-transformer-forward-report /results/transformer-2-candidate-first.json
```

The runner validates these reports as independent evidence before starting its
generation matrix. The harness reuses the trajectory evaluator over
forward-hook snapshots from every entry in `model.blocks`, computes the complete
5-by-5 cross-variant product and all ten same-instance run pairs, and separately
checks the final transformer output. Each report is cryptographically bound to
its component name, resolved component configuration, loaded model manifests,
and fixed tensor inputs, so relabelling one component's report is rejected. The
real model loader and captured forward inputs belong in the remote evaluation
environment; the public harness does not guess checkpoint-specific inputs. Hook
capture is a correctness path and must
not be used for performance timing.

### Component residency

Use `--component-residency COMPONENT=MODE` to choose one runtime mode for each
loaded component:

- `resident` keeps the complete component on the accelerator.
- `component-offload` stores the complete component on CPU between uses.
- `layerwise-offload` streams the component's declared layers from CPU.

`COMPONENT` can be an exact `model_index.json` key or one of `all`, `dit`,
`text_encoder`, `image_encoder`, and `vae`. Exact keys override groups, and
groups override `all`. Existing options such as `--dit-cpu-offload`,
`--text-encoder-cpu-offload`, `--image-encoder-cpu-offload`,
`--vae-cpu-offload`, and `--cpu-offload-components` remain supported. See the
[CLI reference](https://docs.sglang.io/docs/sglang-diffusion/api/cli#component-residency)
for precedence and compatibility details.

### LoRA support

Apply LoRA adapters via `--lora-path`:

```bash
sglang generate \
  --model-path Qwen/Qwen-Image-Edit-2511 \
  --lora-path prithivMLmods/Qwen-Image-Edit-2511-Anime \
  --prompt "Transform into anime." \
  --image-path "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png" \
  --save-output
```

For more usage examples (e.g. OpenAI compatible API, server mode), check the [CLI reference](https://docs.sglang.io/docs/sglang-diffusion/api/cli).

## Contributing

All contributions are welcome. The contribution guide is available [here](https://docs.sglang.io/docs/sglang-diffusion/contributing).

## Acknowledgement

We learnt and reused code from the following projects:

- [FastVideo](https://github.com/hao-ai-lab/FastVideo.git). The major components of this repo are based on a fork of FastVideo on Sept. 24, 2025.
- [xDiT](https://github.com/xdit-project/xDiT). We used the parallelism library from it.
- [diffusers](https://github.com/huggingface/diffusers) We used the pipeline design from it.
