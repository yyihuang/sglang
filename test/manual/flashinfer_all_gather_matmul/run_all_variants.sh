#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

: "${SGLANG_ROOT:?set SGLANG_ROOT}"
: "${RESULT_ROOT:?set RESULT_ROOT}"
: "${MODEL_PATH:?set MODEL_PATH}"
: "${MODEL_RESULT_PATH:?set MODEL_RESULT_PATH}"
: "${WEIGHTS_MANIFEST_PATH:?set WEIGHTS_MANIFEST_PATH}"
: "${GSM8K_PATH:?set GSM8K_PATH}"
: "${SHAREGPT_PATH:?set SHAREGPT_PATH}"
: "${SGLANG_EXPECTED_COMMIT:?set SGLANG_EXPECTED_COMMIT}"
: "${SGLANG_EXPECTED_TREE:?set SGLANG_EXPECTED_TREE}"
: "${FLASHINFER_WHEEL:?set FLASHINFER_WHEEL}"
: "${FLASHINFER_RECEIPT:?set FLASHINFER_RECEIPT}"
: "${FLASHINFER_EXPECTED_COMMIT:?set FLASHINFER_EXPECTED_COMMIT}"
: "${FLASHINFER_EXPECTED_TREE:?set FLASHINFER_EXPECTED_TREE}"
: "${FLASHINFER_EXPECTED_WHEEL_SHA256:?set FLASHINFER_EXPECTED_WHEEL_SHA256}"
: "${FLASHINFER_EXPECTED_API_SIGNATURE:?set FLASHINFER_EXPECTED_API_SIGNATURE}"
: "${FLASHINFER_INSTALL_ROOT:?set FLASHINFER_INSTALL_ROOT}"
: "${SGLANG_CONTAINER_IMAGE:?set SGLANG_CONTAINER_IMAGE}"
readonly harness_dir="$SGLANG_ROOT/test/manual/flashinfer_all_gather_matmul"
if [[ -e "$RESULT_ROOT" || -L "$RESULT_ROOT" ]]; then
  echo "refusing pre-existing result root: $RESULT_ROOT" >&2
  exit 1
fi
mkdir "$RESULT_ROOT"

python3 "$harness_dir/runtime_contract.py" \
  --sglang-root "$SGLANG_ROOT" --sglang-commit "$SGLANG_EXPECTED_COMMIT" \
  --sglang-tree "$SGLANG_EXPECTED_TREE" --flashinfer-wheel "$FLASHINFER_WHEEL" \
  --flashinfer-receipt "$FLASHINFER_RECEIPT" \
  --flashinfer-commit "$FLASHINFER_EXPECTED_COMMIT" \
  --flashinfer-tree "$FLASHINFER_EXPECTED_TREE" \
  --flashinfer-wheel-sha256 "$FLASHINFER_EXPECTED_WHEEL_SHA256" \
  --flashinfer-api-signature "$FLASHINFER_EXPECTED_API_SIGNATURE" \
  --flashinfer-install-root "$FLASHINFER_INSTALL_ROOT" \
  --container-image "$SGLANG_CONTAINER_IMAGE" \
  --expected-cluster oci-hsg-cs-001 \
  --expected-gpu-name-regex 'NVIDIA (GB200|B200)' \
  --output "$RESULT_ROOT/runtime-contract.json"

python3 "$harness_dir/input_contract.py" \
  --model-path "$MODEL_PATH" --model-result "$MODEL_RESULT_PATH" \
  --weights-manifest "$WEIGHTS_MANIFEST_PATH" --gsm8k "$GSM8K_PATH" \
  --sharegpt "$SHAREGPT_PATH" --output "$RESULT_ROOT/input-contract.json"

for variant in native explicit candidate; do
  "$harness_dir/run_variant.sh" "$variant"
done
for variant in explicit candidate; do
  python3 "$harness_dir/compare_fixed_requests.py" \
    --reference "$RESULT_ROOT/native/fixed-requests.json" \
    --candidate "$RESULT_ROOT/$variant/fixed-requests.json" \
    --output "$RESULT_ROOT/$variant/parity-vs-native.json" \
    --required-exact 32 --max-logprob-delta 0.05
done
python3 "$harness_dir/summarize_results.py" \
  --result-root "$RESULT_ROOT" --output "$RESULT_ROOT/summary.json"
printf 'pass\n' > "$RESULT_ROOT/COMPLETE"
