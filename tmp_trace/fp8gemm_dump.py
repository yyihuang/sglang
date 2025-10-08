import json
import os
import uuid
from typing import Dict, Optional, Tuple

import torch
from safetensors.torch import save_file

WORKLOAD_LOG_DIR = "/raid/user_data/yiyanz/fp8_gemm/"
os.makedirs(WORKLOAD_LOG_DIR, exist_ok=True)

def dump_safetensors_tensor_group(tensors: Dict[str, torch.Tensor], prefix: str):
    file_id = uuid.uuid4().hex
    file_path = os.path.join(WORKLOAD_LOG_DIR, f"{prefix}_{file_id}.safetensors")
    save_file(tensors, file_path)
    return file_path, {k: k for k in tensors}

def log_fp8_gemm_inputs(
        A: torch.Tensor,
        B: torch.Tensor,
        A_scale: torch.Tensor,
        B_scale: torch.Tensor,
    ) -> Tuple[str, dict]:
        n = B.shape[0]
        k = A.shape[1]
        definition = f"gemm_fp8_n{n}_k{k}",

        M = A.shape[0]
        tensor_dict: Dict[str, torch.Tensor] = {
            "A": A.cpu(),
            "B": B.cpu(),
            "A_scale": A_scale.cpu(),
            "B_scale": B_scale.cpu(),
        }

        inputs_json: Dict[str, dict] = {
            key: {
                "type": "safetensors",
                "path": None,  # will be filled after dump
                "tensor_key": None,
            }
            for key in tensor_dict
        }

        log_file = f"{definition}.jsonl"
        kv_tensor_file, tensor_keys = dump_safetensors_tensor_group(
            tensor_dict, prefix=definition
        )

        for key in tensor_dict.keys():
            inputs_json[key]["path"] = "./blob/workloads/gemm" + "/" + definition + "/" + os.path.basename(kv_tensor_file)
            inputs_json[key]["tensor_key"] = tensor_keys[key]

        log_entry = {
            "definition": definition,
            "solution": None,
            "workload": {
                "uuid": str(uuid.uuid4()),
                "axes": {
                    "M": M,
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