#!/usr/bin/env bash
#SBATCH --job-name=sglang-agmm-e2e
#SBATCH --partition=batch
#SBATCH --account=coreai_dlcompiler_agent
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=512G
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --no-requeue

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
: "${FLASHINFER_RECEIPT_RELATIVE_PATH:?candidate receipt path below AGMM_SHARED_ROOT}"
: "${SGLANG_EXPECTED_COMMIT:?exact 40-character SGLang commit}"
: "${SGLANG_EXPECTED_TREE:?exact 40-character SGLang tree}"
: "${FLASHINFER_EXPECTED_COMMIT:?exact 40-character FlashInfer commit}"
: "${FLASHINFER_EXPECTED_TREE:?exact 40-character FlashInfer tree}"
: "${FLASHINFER_EXPECTED_WHEEL_SHA256:?exact candidate wheel SHA256}"
: "${FLASHINFER_EXPECTED_API_SIGNATURE:?exact public API signature}"
: "${SGLANG_CONTAINER_IMAGE:?digest-pinned container image}"

if [[ "$SGLANG_CONTAINER_IMAGE" != *@sha256:* ]]; then
  echo "SGLANG_CONTAINER_IMAGE must be pinned by sha256 digest" >&2
  exit 1
fi
for identity in "$SGLANG_EXPECTED_COMMIT" "$SGLANG_EXPECTED_TREE" \
  "$FLASHINFER_EXPECTED_COMMIT" "$FLASHINFER_EXPECTED_TREE"; do
  if [[ ! "$identity" =~ ^[0-9a-f]{40}$ ]]; then
    echo "invalid 40-character source identity: $identity" >&2
    exit 1
  fi
done
if [[ ! "$FLASHINFER_EXPECTED_WHEEL_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "invalid FlashInfer wheel SHA256" >&2
  exit 1
fi
for relative in "$MODEL_RELATIVE_PATH" "$MODEL_RESULT_RELATIVE_PATH" \
  "$WEIGHTS_MANIFEST_RELATIVE_PATH" "$GSM8K_RELATIVE_PATH" \
  "$SHAREGPT_RELATIVE_PATH" "$RESULT_RELATIVE_PATH" \
  "$FLASHINFER_WHEEL_RELATIVE_PATH" "$FLASHINFER_RECEIPT_RELATIVE_PATH"; do
  if [[ "$relative" == /* || "$relative" == .. || "$relative" == ../* || \
        "$relative" == */../* || "$relative" == */.. ]]; then
    echo "relative input escapes AGMM_SHARED_ROOT: $relative" >&2
    exit 1
  fi
done

readonly wheel="$AGMM_SHARED_ROOT/$FLASHINFER_WHEEL_RELATIVE_PATH"
readonly flashinfer_receipt="$AGMM_SHARED_ROOT/$FLASHINFER_RECEIPT_RELATIVE_PATH"
readonly model="$AGMM_SHARED_ROOT/$MODEL_RELATIVE_PATH"
readonly model_result="$AGMM_SHARED_ROOT/$MODEL_RESULT_RELATIVE_PATH"
readonly weights_manifest="$AGMM_SHARED_ROOT/$WEIGHTS_MANIFEST_RELATIVE_PATH"
readonly gsm8k="$AGMM_SHARED_ROOT/$GSM8K_RELATIVE_PATH"
readonly sharegpt="$AGMM_SHARED_ROOT/$SHAREGPT_RELATIVE_PATH"
readonly result_root="$AGMM_SHARED_ROOT/$RESULT_RELATIVE_PATH"
for path in "$wheel" "$flashinfer_receipt" "$model_result" \
  "$weights_manifest" "$gsm8k" "$sharegpt"; do
  test -f "$path"
done
test -d "$model"
test "$(git -C "$SGLANG_CHECKOUT" rev-parse --is-inside-work-tree)" = true
test "$(git -C "$SGLANG_CHECKOUT" rev-parse HEAD)" = "$SGLANG_EXPECTED_COMMIT"
test "$(git -C "$SGLANG_CHECKOUT" rev-parse HEAD^{tree})" = "$SGLANG_EXPECTED_TREE"
test -z "$(git -C "$SGLANG_CHECKOUT" status --porcelain=v1)"
test ! -e "$result_root"
test ! -L "$result_root"
IFS='|' read -r task_scheduler_submit_time task_allocation_start_time < <(
  squeue --noheader --jobs "$SLURM_JOB_ID" --format='%V|%S'
)
test -n "$task_scheduler_submit_time"
test -n "$task_allocation_start_time"

srun \
  --nodes=1 --ntasks=1 --cpus-per-task=32 --gres=gpu:4 \
  --container-image="$SGLANG_CONTAINER_IMAGE" \
  --no-container-mount-home \
  --container-mounts="$AGMM_SHARED_ROOT:$AGMM_SHARED_ROOT,$SGLANG_CHECKOUT:$SGLANG_CHECKOUT:ro" \
  --container-workdir="$SGLANG_CHECKOUT" \
  env \
    AGMM_SHARED_ROOT="$AGMM_SHARED_ROOT" \
    SGLANG_CHECKOUT="$SGLANG_CHECKOUT" \
    MODEL_PATH="$model" \
    MODEL_RESULT_PATH="$model_result" \
    WEIGHTS_MANIFEST_PATH="$weights_manifest" \
    GSM8K_PATH="$gsm8k" \
    SHAREGPT_PATH="$sharegpt" \
    RESULT_ROOT="$result_root" \
    FLASHINFER_WHEEL="$wheel" \
    FLASHINFER_RECEIPT="$flashinfer_receipt" \
    SGLANG_EXPECTED_COMMIT="$SGLANG_EXPECTED_COMMIT" \
    SGLANG_EXPECTED_TREE="$SGLANG_EXPECTED_TREE" \
    FLASHINFER_EXPECTED_COMMIT="$FLASHINFER_EXPECTED_COMMIT" \
    FLASHINFER_EXPECTED_TREE="$FLASHINFER_EXPECTED_TREE" \
    FLASHINFER_EXPECTED_WHEEL_SHA256="$FLASHINFER_EXPECTED_WHEEL_SHA256" \
    FLASHINFER_EXPECTED_API_SIGNATURE="$FLASHINFER_EXPECTED_API_SIGNATURE" \
    FLASHINFER_DISABLE_VERSION_CHECK=1 \
    SGLANG_CONTAINER_IMAGE="$SGLANG_CONTAINER_IMAGE" \
    AGMM_EXPECTED_KERNEL_SYMBOL=kernel_cake_blackwell_all_gather_matmul_bfloat16_ws4 \
    TASK_SCHEDULER_SUBMIT_TIME="$task_scheduler_submit_time" \
    TASK_ALLOCATION_START_TIME="$task_allocation_start_time" \
    PYTHONNOUSERSITE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
  bash --noprofile --norc -c '
set -euo pipefail
readonly runtime_site=$(mktemp -d "/tmp/flashinfer-runtime-${SLURM_JOB_ID}.XXXXXX")
trap '\''rm -rf -- "$runtime_site"'\'' EXIT
python3 -m pip install --disable-pip-version-check --target "$runtime_site" \
  --force-reinstall --no-deps "$FLASHINFER_WHEEL"
export PYTHONPATH="$runtime_site:$SGLANG_CHECKOUT/python"
export FLASHINFER_INSTALL_ROOT="$runtime_site"
export SGLANG_ROOT="$SGLANG_CHECKOUT"
"$SGLANG_CHECKOUT/test/manual/flashinfer_all_gather_matmul/run_all_variants.sh"
'
