import os
import uuid
import json
from typing import Optional, Tuple, Dict

import torch
from safetensors.torch import save_file

WORKLOAD_LOG_DIR = "tmp_trace/dump_results_fused_moe"
os.makedirs(WORKLOAD_LOG_DIR, exist_ok=True)


def dump_safetensors_tensor_group(tensors: Dict[str, torch.Tensor], prefix: str):
    file_id = uuid.uuid4().hex
    file_path = os.path.join(WORKLOAD_LOG_DIR, f"{prefix}_{file_id}.safetensors")
    save_file(tensors, file_path)
    return file_path, {k: k for k in tensors}


def log_fused_moe_inputs(
    routing_logits: torch.Tensor,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    routing_bias: Optional[torch.Tensor],
    local_expert_offset: int,
    routed_scaling_factor: Optional[float] = None,
) -> Tuple[str, dict]:
    seq_len = routing_logits.shape[0]
    definition_name = "fused_moe_n" + str(seq_len)

    tensor_dict: Dict[str, torch.Tensor] = {
        "routing_logits": routing_logits.cpu(),
        "hidden_states": hidden_states.cpu(),
        "hidden_states_scale": hidden_states_scale.cpu(),
    }
    inputs_json: Dict[str, dict] = {
        key: {
            "type": "safetensors",
            "path": None,  # will be filled after dump
            "tensor_key": None,
        }
        for key in tensor_dict
    }

    if routing_bias is not None:
        tensor_dict["routing_bias"] = routing_bias.cpu()
        inputs_json["routing_bias"] = {
            "type": "safetensors",
            "path": None,
            "tensor_key": None,
        }

    log_file = f"{definition_name}.jsonl"
    kv_tensor_file, tensor_keys = dump_safetensors_tensor_group(
        tensor_dict, prefix=definition_name
    )

    for key in tensor_dict.keys():
        inputs_json[key]["path"] = os.path.basename(kv_tensor_file)
        inputs_json[key]["tensor_key"] = tensor_keys[key]

    inputs_json["local_expert_offset"] = {
        "type": "scalar",
        "value": int(local_expert_offset),
    }

    inputs_json["routed_scaling_factor"] = {
        "type": "scalar",
        "value": None
        if routed_scaling_factor is None
        else float(routed_scaling_factor),
    }

    log_entry = {
        "definition": definition_name,
        "solution": None,
        "workload": {
            "uuid": str(uuid.uuid4()),
            "axes": {
                "seq_len": seq_len,
            },
            "inputs": inputs_json,
        },
        "evaluation": None,
    }

    os.makedirs(WORKLOAD_LOG_DIR, exist_ok=True)
    path = os.path.join(WORKLOAD_LOG_DIR, log_file)
    with open(path, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    return path, log_entry
