import os
import uuid
import json
from typing import Optional, Tuple, List
import torch

WORKLOAD_LOG_DIR = # TO_ADD_YOUR_PATH
os.makedirs(WORKLOAD_LOG_DIR, exist_ok=True)
from safetensors.torch import save_file
def dump_safetensors_tensor_group(tensors: dict, prefix: str):
    file_id = uuid.uuid4().hex
    file_path = os.path.join(WORKLOAD_LOG_DIR, f"{prefix}_{file_id}.safetensors")
    save_file(tensors, file_path)
    return file_path, {k: k for k in tensors}


def log_sampling(logits: torch.Tensor, top_ks: Optional[List[int]] = None, top_ps: Optional[List[float]] = None) -> Tuple[str, dict]:
    definition_name = "sampling_from_probs_v" + str(logits.shape[1])
    bs = logits.shape[0]

    tensor_dict = {"probs": logits.cpu()}
    inputs_json = {
        "probs": {
            "type": "safetensors",
            "path": None,  # will be filled after dump
            "tensor_key": None,
        }
    }
    
    if top_ps is not None:
        p_val = top_ps[0]
        tensor_dict["top_p"] = torch.full((bs,), float(p_val), dtype=torch.float32)
        inputs_json["top_p"] = {
            "type": "safetensors",
            "path": None,
            "tensor_key": None,
        }
        definition_name = "top_p_" + definition_name
    if top_ks is not None:
        k_val = top_ks[0]
        tensor_dict["top_k"] = torch.full((bs,), int(k_val), dtype=torch.int32)
        inputs_json["top_k"] = {
            "type": "safetensors",
            "path": None,
            "tensor_key": None,
        }
        definition_name = "top_k_" + definition_name

    log_file = f"{definition_name}.jsonl"
    kv_tensor_file, tensor_keys = dump_safetensors_tensor_group(tensor_dict, prefix=definition_name)

    # Fill in paths and tensorKeys
    for k in tensor_dict.keys():
        inputs_json[k]["path"] = os.path.basename(kv_tensor_file)
        inputs_json[k]["tensor_key"] = tensor_keys[k]

    log_entry = {
        "definition": definition_name,
        "solution": None,
        "workload": {
            "uuid": str(uuid.uuid4()),
            "axes": {
                "batch_size": bs,
            },
            "inputs": inputs_json,
        },
        "evaluation": None
    }

    os.makedirs(WORKLOAD_LOG_DIR, exist_ok=True)
    path = os.path.join(WORKLOAD_LOG_DIR, log_file)
    with open(path, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    return path, log_entry

# later in Sampler class, add log_sampling(logits=logits, top_ks=sampling_info.top_ks, top_ps=sampling_info.top_ps) in forward function


