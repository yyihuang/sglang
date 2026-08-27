<div align="center"  style="display:block; margin:auto;">
<img src=https://github.com/lm-sys/lm-sys.github.io/releases/download/test/sgl-diffusion-logo.png width="80%"/>
</div>

**SGLang diffusion is an inference framework for accelerated image/video generation.**

SGLang diffusion features an end-to-end unified pipeline for accelerating diffusion models. It is designed to be modular and extensible, allowing users to easily add new models and optimizations.

## Key Features

SGLang Diffusion has the following features:
  - Broad model support: Wan series, FastWan series, Hunyuan, Qwen-Image, Qwen-Image-Edit, Flux, Z-Image, GLM-Image
  - Fast inference speed: enpowered by highly optimized kernel from sgl-kernel and efficient scheduler loop
  - Ease of use: OpenAI-compatible api, CLI, and python sdk support
  - Multi-platform support: NVIDIA GPUs (H100, H200, A100, B200, 4090) and AMD GPUs (MI300X, MI325X)

### AMD/ROCm Support

SGLang Diffusion supports AMD Instinct GPUs through ROCm. On AMD platforms, we use the Triton attention backend and leverage AITER kernels for optimized layernorm and other operations. See the [ROCm installation guide](https://github.com/sgl-project/sglang/tree/main/python/sglang/multimodal_gen/docs/install_rocm.md) for setup instructions.

## Getting Started

```bash
uv pip install 'sglang[diffusion]' --prerelease=allow
```

For more installation methods (e.g. pypi, uv, docker), check [install.md](https://github.com/sgl-project/sglang/tree/main/python/sglang/multimodal_gen/docs/install.md). ROCm/AMD users should follow the [ROCm quickstart](https://github.com/sgl-project/sglang/tree/main/python/sglang/multimodal_gen/docs/install_rocm.md) that includes the additional kernel builds and attention backend settings we validated on MI300X.


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
    --text-encoder-cpu-offload --pin-cpu-memory \
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
model-level correctness claim. The `wan_hybrid_min_timestep` and
`wan_hybrid_layer_indices` backend options are diagnostic gates. A run is not a
valid hybrid qualification unless its reported `wan_hybrid_hit_count` is
greater than zero.

Use `compare_diffusion_trajectory_similarity` for model-level qualification.
The tool requires at least two warmup runs and five measured runs. Correctness
captures every trajectory step and evaluates every same-variant and
cross-variant run pair. Performance disables trajectory capture, executes both
reference-first and candidate-first orders, and passes only when both median
speedups are at least 1.0 and every measured candidate run reports a positive
backend hit count. Every successful hybrid call also records and validates the
exact serving boundary (`B=1, S=4800, H=40, D=128`, NHD, noncausal, raw
post-RoPE BF16 Q/K/V, and caller-owned BF16 output storage). A hit count without
the corresponding per-call boundary record is not qualification evidence.

The qualification runner builds fixed single-block, full-transformer, and
generation matrices without depending on a particular cluster layout. Pass the
staged public revisions explicitly so the manifest records what was measured:

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

A correctness invocation runs both execution orders as separate generation
trajectory reports. A performance invocation runs one comparison command with
`--run-order both`; it disables trajectory capture and records both orders in
the same report. `single-block` selects block zero through
`wan_hybrid_layer_indices`; `full-transformer` selects the primary Wan
transformer component; and `generation` enables every eligible self-attention
layer.

For an independent transformer single-forward check, first capture the exact
keyword arguments from one real singleton serving request:

```bash
python -m sglang.multimodal_gen.tools.capture_wan_transformer_inputs \
  --model-path /models/wan \
  --output-dir /results/wan-inputs \
  --output-index-json /results/wan-inputs/index.json \
  --prompt "qualification prompt" --seed 4254 \
  --width 1280 --height 720 --num-frames 81 \
  --num-inference-steps 30 --guidance-scale 5.0 --guidance-scale-2 5.0 \
  --component transformer --component transformer_2
```

Each worker manifest binds the request, sampling parameters, model/component
identity, step/timestep/CFG branch, and CPU tensor artifact. Reports retain the
capture artifact, canonical fixed-input, and canonical invocation digests;
these digests are not interchangeable. Correctness uses separately configured
reference and candidate instances and runs both explicit execution orders:

```bash
python -m sglang.multimodal_gen.tools.run_wan_transformer_forward_report \
  --capture-manifest /results/wan-inputs/<transformer-manifest>.json \
  --run-order reference-first \
  --output-json /results/transformer-reference-first.json
```

Repeat for `candidate-first` and `transformer_2`, then provide all four reports
to a `full-transformer` qualification. The harness reuses the trajectory
evaluator over snapshots from every `model.blocks` entry, computes the complete
5-by-5 cross-variant product and all ten same-instance run pairs, and separately
checks the final transformer output. Direct performance instead prepares FA
once and switches the same candidate model instance request-locally between FA
and its construction-default `wan_hybrid` implementation. Both orders reuse the
same model, fixed input object, CUDA device, process, and stream. Hook capture
is a correctness path and must not be used for performance timing.

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

For more usage examples (e.g. OpenAI compatible API, server mode), check [cli.md](https://github.com/sgl-project/sglang/tree/main/python/sglang/multimodal_gen/docs/cli.md).

## Contributing

All contributions are welcome. The contribution guide is available [here](https://github.com/sgl-project/sglang/tree/main/python/sglang/multimodal_gen/docs/contributing.md).

## Acknowledgement

We learnt and reused code from the following projects:

- [FastVideo](https://github.com/hao-ai-lab/FastVideo.git). The major components of this repo are based on a fork of FastVideo on Sept. 24, 2025.
- [xDiT](https://github.com/xdit-project/xDiT). We used the parallelism library from it.
- [diffusers](https://github.com/huggingface/diffusers) We used the pipeline design from it.
