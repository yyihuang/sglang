#!/usr/bin/env bash
set -euo pipefail

: "${SGLANG_ROOT:?set SGLANG_ROOT}"
: "${RESULT_ROOT:?set RESULT_ROOT}"
: "${MODEL_PATH:?set MODEL_PATH}"
: "${MODEL_RESULT_PATH:?set MODEL_RESULT_PATH}"
: "${WEIGHTS_MANIFEST_PATH:?set WEIGHTS_MANIFEST_PATH}"
: "${GSM8K_PATH:?set GSM8K_PATH}"
: "${SHAREGPT_PATH:?set SHAREGPT_PATH}"
readonly harness_dir="$SGLANG_ROOT/test/manual/flashinfer_all_gather_matmul"
mkdir -p "$RESULT_ROOT"

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
