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

### Cake NVFP4 attention for Wan

On B200/GB200 (`sm_100`) and B300/GB300 (`sm_103`), a FlashInfer build that
exports `flashinfer.nvfp4_attention` can run the dense self-attention layers of
Wan's ModelOpt NVFP4 checkpoint through the Cake backend:

```bash
sglang generate \
  --model-path nvidia/Wan2.2-T2V-A14B-Diffusers-NVFP4 \
  --attention-backend cake_nvfp4 \
  --prompt "A curious raccoon walks through a sunlit forest" \
  --save-output
```

This backend is intentionally fail-closed: it accepts BF16, noncausal dense
self-attention with equal Q/K/V shapes and head dimension 128. Wan
cross-attention continues to use the normal dense backend because its query and
KV sequence lengths differ. Packed-varlen, masks, GQA/MQA, and ring attention
are not supported. Ulysses-only sequence parallelism is supported when each
rank retains a valid local head count.

This integration is experimental and is not a production quality claim. The
Wan path fuses RMSNorm, RoPE, Q-centering, and FP4 packing, reuses its packed
workspace and caller-owned output, and supplies the matching FP32 QK-logit
correction to FlashInfer. Complete diffusion trajectories and generated frames
must still be qualified against the dense backend: isolated attention accuracy
is insufficient for this model.

The current B200 qualification uses the exact ModelOpt checkpoint above at
640x384, 17 frames, 12 steps, seed 0, TP1/SP1, two warmups, and five measured
requests. BF16 FlashAttention averages 2.718266 seconds and all-layer corrected
Cake averages 2.739915 seconds (0.9921x). The final denoising-trajectory cosine
is 0.6591; all-frame cosine is 0.7338 and PSNR is 11.87 dB. Even restricting
Cake to one transformer block in the final denoising step leaves final-frame
PSNR near 12.15 dB. The error amplification is therefore a full-model
sensitivity to FP4 attention, not an unaccounted wrapper allocation or missing
QK correction.

Cake remains an explicit opt-in and the production route stays on FA. The
`cake_nvfp4_min_timestep` and `cake_nvfp4_layer_indices` backend options are
diagnostic gates; a configuration that produces zero Cake calls is not a
successful Cake serving integration.

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
