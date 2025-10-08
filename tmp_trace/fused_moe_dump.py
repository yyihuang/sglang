import json
import os
import uuid
from typing import Dict, Optional, Tuple

import torch
from safetensors.torch import save_file

WORKLOAD_LOG_DIR = "/raid/user_data/YOUR_ID/fused_moe/"
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
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    local_expert_offset: int,
    routed_scaling_factor: float,
) -> Tuple[str, dict]:
    seq_len = routing_logits.shape[0]
    definition_name = "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048"

    tensor_dict: Dict[str, torch.Tensor] = {
        "routing_logits": routing_logits.cpu(),
        "hidden_states": hidden_states.cpu(),
        "hidden_states_scale": hidden_states_scale.cpu(),
        "routing_bias": routing_bias.cpu(),
        "gemm1_weights": gemm1_weights.cpu(),
        "gemm1_weights_scale": gemm1_weights_scale.cpu(),
        "gemm2_weights": gemm2_weights.cpu(),
        "gemm2_weights_scale": gemm2_weights_scale.cpu(),
    }
    # for key, tensor in tensor_dict.items():
    #     if tensor is not None:
    #         print(f"{key}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")
    #     else:
    #         print(f"{key}: None")
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
        inputs_json[key]["path"] = "./blob/workloads/moe" + "/" + definition_name + "/" + os.path.basename(kv_tensor_file)
        inputs_json[key]["tensor_key"] = tensor_keys[key]

    inputs_json["local_expert_offset"] = {
        "type": "scalar",
        "value": int(local_expert_offset),
    }

    inputs_json["routed_scaling_factor"] = {
        "type": "scalar",
        "value": (
            None if routed_scaling_factor is None else float(routed_scaling_factor)
        ),
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
