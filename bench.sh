# 1. Download ShareGPT if you haven't already
# wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

# 2. Run the benchmark sweep
# "Batch size" here is simulated by "--max-concurrency"
for BS in 1 2 3 4 5 6 7 8 10 12 14 16 18 20 22 24 26 28 30 32 40 44 48 64 128; do
    echo "======================================================="
    echo "Running ShareGPT benchmark with Concurrency: $BS"
    echo "======================================================="
    
    python3 -m sglang.bench_serving \
        --backend sglang \
        --dataset-name sharegpt \
        --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json \
        --host 127.0.0.1 \
        --port 30000 \
        --num-prompts 1 \
        --max-concurrency $BS \
        --tokenizer Qwen/Qwen3-Next-80B-A3B-Instruct
        
    # Optional: Sleep briefly to let the server cool down/clear queue
    sleep 5
done
