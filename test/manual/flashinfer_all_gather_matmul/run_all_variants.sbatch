#!/usr/bin/env bash
#SBATCH --job-name=sglang-agmm-e2e
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --time=02:30:00

set -euo pipefail

: "${AGMM_SHARED_ROOT:?host path containing models, datasets, manifests, and results}"
: "${SGLANG_CHECKOUT:?host path to this exact SGLang checkout}"
: "${MODEL_RELATIVE_PATH:?path below AGMM_SHARED_ROOT}"
: "${MODEL_RESULT_RELATIVE_PATH:?path below AGMM_SHARED_ROOT}"
: "${WEIGHTS_MANIFEST_RELATIVE_PATH:?path below AGMM_SHARED_ROOT}"
: "${GSM8K_RELATIVE_PATH:?path below AGMM_SHARED_ROOT}"
: "${SHAREGPT_RELATIVE_PATH:?path below AGMM_SHARED_ROOT}"
: "${RESULT_RELATIVE_PATH:?path below AGMM_SHARED_ROOT}"
: "${FLASHINFER_WHEEL_RELATIVE_PATH:?candidate wheel path below AGMM_SHARED_ROOT}"

readonly image=${SGLANG_CONTAINER_IMAGE:-docker.io#lmsysorg/sglang:dev}
srun \
  --container-image="$image" \
  --container-mounts="$AGMM_SHARED_ROOT:/shared,$SGLANG_CHECKOUT:/workspace/sglang" \
  --container-workdir=/workspace/sglang \
  bash --noprofile --norc -c '
set -euo pipefail
python3 -m pip install --no-deps "/shared/$FLASHINFER_WHEEL_RELATIVE_PATH"
export SGLANG_ROOT=/workspace/sglang
export MODEL_PATH="/shared/$MODEL_RELATIVE_PATH"
export MODEL_RESULT_PATH="/shared/$MODEL_RESULT_RELATIVE_PATH"
export WEIGHTS_MANIFEST_PATH="/shared/$WEIGHTS_MANIFEST_RELATIVE_PATH"
export GSM8K_PATH="/shared/$GSM8K_RELATIVE_PATH"
export SHAREGPT_PATH="/shared/$SHAREGPT_RELATIVE_PATH"
export RESULT_ROOT="/shared/$RESULT_RELATIVE_PATH"
export AGMM_EXPECTED_KERNEL_REGEX=kernel_cake_blackwell_all_gather_matmul_bfloat16_ws4
exec /workspace/sglang/test/manual/flashinfer_all_gather_matmul/run_all_variants.sh
'
