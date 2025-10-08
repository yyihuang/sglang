import json
import random
import os
import shutil
from collections import defaultdict

YOUR_ID = "yiyanz"

# Assuming the JSONL file is in the current directory
filename = f"/raid/user_data/{YOUR_ID}/fused_moe/moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048.jsonl"

# Group entries by sequence length
groups = defaultdict(list)
with open(filename, 'r') as f:
    for line in f:
        entry = json.loads(line.strip())
        seq_len = entry['workload']['axes']['seq_len']
        groups[seq_len].append(entry)

# Calculate total lines
total_lines = sum(len(entries) for entries in groups.values())
target_total = 50

kept = []

for seq_len, entries in groups.items():
    # # Calculate proportional keep count
    # proportion = len(entries) / total_lines
    # num_keep = max(1, round(target_total * proportion))
    num_keep = 1
    num_keep = min(num_keep, len(entries))  # Don't keep more than available
    
    # Randomly shuffle and select
    random.shuffle(entries)
    kept.extend(entries[:num_keep])

# kept = []
# removed = []
# n = 4

# for seq_len, entries in groups.items():
#     if len(entries) > n:
#         # Randomly shuffle and select n to keep
#         random.shuffle(entries)
#         kept.extend(entries[:n])
#     else:
#         kept.extend(entries)

# Copy the corresponding tensor files for removed entries to /blob/
for entry in kept:
    # Assuming all inputs share the same path (the safetensors file)
    tensor_path = entry['workload']['inputs']['routing_logits']['path']
    tensor_basename = os.path.basename(tensor_path)
    full_tensor_path = os.path.join(f"/raid/user_data/{YOUR_ID}/fused_moe", tensor_basename)
    blob_dir = f"/raid/user_data/{YOUR_ID}/blob/"
    os.makedirs(blob_dir, exist_ok=True)
    shutil.copy(full_tensor_path, os.path.join(blob_dir, tensor_basename))

# Write the kept entries to a new JSONL file
output_filename = filename + "_filtered"
with open(output_filename, 'w') as f:
    for entry in kept:
        f.write(json.dumps(entry) + '\n')

print(f"Processed {filename}. Kept entries written to {output_filename}. Removed tensors copied to /blob/")