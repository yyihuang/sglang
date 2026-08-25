#!/usr/bin/env bash
set -euo pipefail

readonly variant=${1:?usage: run_variant.sh native|explicit|candidate}
case "$variant" in
  native|explicit|candidate) ;;
  *) echo "invalid variant: $variant" >&2; exit 2 ;;
esac

: "${SGLANG_ROOT:?set SGLANG_ROOT}"
: "${MODEL_PATH:?set MODEL_PATH}"
: "${RESULT_ROOT:?set RESULT_ROOT}"
: "${SHAREGPT_PATH:?set SHAREGPT_PATH}"
: "${GSM8K_PATH:?set GSM8K_PATH}"
: "${SGLANG_EXPECTED_COMMIT:?set SGLANG_EXPECTED_COMMIT}"
: "${SGLANG_EXPECTED_TREE:?set SGLANG_EXPECTED_TREE}"

readonly harness_dir="$SGLANG_ROOT/test/manual/flashinfer_all_gather_matmul"
readonly result_dir="$RESULT_ROOT/$variant"
readonly port=${SGLANG_PORT:-30000}
readonly base_url="http://127.0.0.1:$port"
readonly model_revision=1605565b47bb9346c5515c34102e054115b4f98b
readonly kernel_regex=${AGMM_EXPECTED_KERNEL_REGEX:-'kernel_cake_blackwell_all_gather_matmul_bfloat16_ws4'}
if [[ -e "$result_dir" || -L "$result_dir" ]]; then
  echo "refusing pre-existing arm artifact: $result_dir" >&2
  exit 1
fi
mkdir "$result_dir"
mkdir "$result_dir/profile"

export PYTHONPATH="$SGLANG_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
unset DUMPER_SOURCE_PATCHER_CONFIG DUMPER_SERVER_PORT \
  DUMPER_NON_INTRUSIVE_MODE DUMPER_ENABLE AGMM_EXPERIMENT_VARIANT
if [[ "$variant" != native ]]; then
  export PYTHONPATH="$harness_dir:$PYTHONPATH"
  export DUMPER_SOURCE_PATCHER_CONFIG="$harness_dir/source-patch.yaml"
  # Source patches are applied only when dumper.may_enable is true. Reuse mode
  # activates patching without a standalone HTTP listener; mode=off avoids
  # installing non-intrusive tensor hooks in either experimental arm.
  export DUMPER_SERVER_PORT=reuse
  export DUMPER_NON_INTRUSIVE_MODE=off
  export AGMM_EXPERIMENT_VARIANT="$variant"
  export AGMM_EXPERIMENT_ARTIFACT_DIR="$result_dir"
  export AGMM_EXPERIMENT_MIN_FULL_TOKENS=512
  export AGMM_EXPERIMENT_VALIDATE_CALLS=80
fi

readonly physical_start=$(date +%s)
date --iso-8601=seconds > "$result_dir/start-time.txt"
actual_sglang_commit=$(git -C "$SGLANG_ROOT" rev-parse HEAD)
actual_sglang_tree=$(git -C "$SGLANG_ROOT" rev-parse HEAD^{tree})
actual_sglang_status=$(git -C "$SGLANG_ROOT" status --porcelain=v1)
if [[ "$actual_sglang_commit" != "$SGLANG_EXPECTED_COMMIT" || \
      "$actual_sglang_tree" != "$SGLANG_EXPECTED_TREE" || \
      -n "$actual_sglang_status" ]]; then
  echo "SGLang source identity mismatch" >&2
  exit 1
fi
printf '%s\n' "$actual_sglang_commit" > "$result_dir/sglang-commit.txt"
printf '%s\n' "$actual_sglang_tree" > "$result_dir/sglang-tree.txt"
nvidia-smi --query-gpu=index,name,uuid,compute_cap,pci.bus_id,memory.total --format=csv \
  > "$result_dir/gpus.csv"
python3 - "$result_dir/environment.json" "$variant" "$model_revision" <<'PY'
import importlib.metadata
import json
import platform
import sys

output, variant, revision = sys.argv[1:]
packages = {}
for name in ("sglang", "flashinfer-python", "torch", "transformers"):
    try:
        packages[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        packages[name] = None
with open(output, "w") as stream:
    json.dump(
        {
            "variant": variant,
            "model_repo": "meta-llama/Llama-3.1-70B-Instruct",
            "model_revision": revision,
            "platform": platform.platform(),
            "packages": packages,
        },
        stream,
        indent=2,
        sort_keys=True,
    )
    stream.write("\n")
PY

server_pid=
stop_server() {
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill -TERM -- -"$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap stop_server EXIT

PYTHONUNBUFFERED=1 setsid python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --tp-size 4 \
  --dtype bfloat16 \
  --attention-backend flashinfer \
  --disable-cuda-graph \
  --disable-radix-cache \
  --disable-overlap-schedule \
  --chunked-prefill-size 8192 \
  --random-seed 20260825 \
  --max-running-requests 256 \
  --mem-fraction-static 0.80 \
  --host 127.0.0.1 \
  --port "$port" \
  > "$result_dir/server.log" 2>&1 &
server_pid=$!

readonly ready_start=$(date +%s)
for _ in $(seq 1 900); do
  if curl -fsS "$base_url/health" >/dev/null; then
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    tail -n 120 "$result_dir/server.log" >&2
    exit 1
  fi
  sleep 2
done
curl -fsS "$base_url/health" >/dev/null
readonly ready_end=$(date +%s)
echo $((ready_end - ready_start)) > "$result_dir/server-ready-seconds.txt"
readonly patch_marker='[source_patcher] patching sglang.srt.layers.linear.ColumnParallelLinear.forward'
if [[ "$variant" == native ]]; then
  if grep -Fq "$patch_marker" "$result_dir/server.log"; then
    echo "native arm unexpectedly loaded the source patch" >&2
    exit 1
  fi
else
  patch_count=$(grep -Fc "$patch_marker" "$result_dir/server.log" || true)
  if [[ "$patch_count" -ne 4 ]]; then
    echo "expected source patch activation on four TP ranks, found $patch_count" >&2
    exit 1
  fi
  grep -F "$patch_marker" "$result_dir/server.log" > "$result_dir/source-patch-evidence.txt"
fi

python3 "$harness_dir/fixed_requests.py" \
  --base-url "$base_url" --gsm8k "$GSM8K_PATH" \
  --output "$result_dir/fixed-requests.json"

model_key=${MODEL_PATH//\//_}
rm -f "/tmp/gsm8k_${model_key}.json" "/tmp/gsm8k_${model_key}.html"
python3 -m sglang.test.run_eval \
  --base-url "$base_url" --model "$MODEL_PATH" --eval-name gsm8k \
  --api generate --num-examples 500 --num-threads 64 --num-shots 5 \
  --gsm8k-data-path "$GSM8K_PATH" --temperature 0 --top-p 1 \
  --max-tokens 512 > "$result_dir/gsm8k.log" 2>&1
cp "/tmp/gsm8k_${model_key}.json" "$result_dir/gsm8k.json"
cp "/tmp/gsm8k_${model_key}.html" "$result_dir/gsm8k.html"
python3 "$harness_dir/verify_gsm8k.py" \
  --metrics "$result_dir/gsm8k.json" --report "$result_dir/gsm8k.html" \
  --expected-examples 500 --output "$result_dir/gsm8k-evidence.json"

bench_common=(
  --backend sglang --base-url "$base_url" --model "$MODEL_PATH"
  --dataset-name random --dataset-path "$SHAREGPT_PATH"
  --random-input-len 4096 --random-output-len 128 --random-range-ratio 0
  --request-rate inf --max-concurrency 64 --seed 20260825 --temperature 0
  --disable-tqdm --output-details
)
python3 -m sglang.bench_serving "${bench_common[@]}" \
  --num-prompts 32 --output-file "$result_dir/serving-warmup.jsonl" \
  > "$result_dir/serving-warmup.log" 2>&1
for repetition in 1 2 3; do
  python3 -m sglang.bench_serving "${bench_common[@]}" \
    --num-prompts 256 --output-file "$result_dir/serving-${repetition}.jsonl" \
    > "$result_dir/serving-${repetition}.log" 2>&1
done

curl -fsS -X POST "$base_url/start_profile" \
  -H 'Content-Type: application/json' \
  -d "{\"activities\":[\"GPU\"],\"output_dir\":\"$result_dir/profile\",\"profile_prefix\":\"agmm-$variant\"}" \
  > "$result_dir/start-profile.txt"
python3 -m sglang.bench_serving "${bench_common[@]}" \
  --num-prompts 8 --random-output-len 4 \
  --output-file "$result_dir/profile-serving.jsonl" \
  > "$result_dir/profile-serving.log" 2>&1
curl -fsS -X POST "$base_url/stop_profile" > "$result_dir/stop-profile.txt"

stop_server
server_pid=
if [[ "$variant" == native ]]; then
  if compgen -G "$result_dir/agmm-route-rank*.json" >/dev/null; then
    echo "native arm unexpectedly emitted AGMM route counters" >&2
    exit 1
  fi
else
  python3 "$harness_dir/verify_route.py" --variant "$variant" \
    --artifact-dir "$result_dir" --output "$result_dir/route-evidence.json"
fi
if [[ "$variant" == candidate ]]; then
  python3 "$harness_dir/summarize_trace.py" --trace-dir "$result_dir/profile" \
    --kernel-regex "$kernel_regex" --output "$result_dir/cake-kernel-evidence.json"
fi

readonly physical_end=$(date +%s)
echo $((physical_end - physical_start)) > "$result_dir/physical-seconds.txt"
date --iso-8601=seconds > "$result_dir/end-time.txt"
printf 'pass\n' > "$result_dir/COMPLETE"
