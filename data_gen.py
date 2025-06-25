import argparse
import asyncio
import os
import queue
import re
import threading

import numpy as np
import torch
from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="sglang data gen")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=20)
    parser.add_argument("--index", type=int, default=1)
    parser.add_argument("--gpu_index", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--outdir", type=str, default="/root/.cache/hidden_states_dump")
    parser.add_argument(
        "--model_name", type=str, default="meta-llama/Llama-4-Scout-17B-16E-Instruct"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["sharegpt", "ultrachat", "mixture_of_thoughts"],
        default="sharegpt",
    )
    parser.add_argument(
        "--num-consumers",
        type=int,
        default=4,
        help="Number of consumer workers for saving data.",
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.gpu_index))
    MAX_TOKEN_LENGTH = 2048

    # ------------------------ 1. Dataset ------------------------
    # This step converts the dataset into a standard messages format
    if args.dataset == "sharegpt":
        dataset = load_dataset("Aeala/ShareGPT_Vicuna_unfiltered", split="train")
    elif args.dataset == "ultrachat":
        dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    elif args.dataset == "mixture_of_thoughts":
        dataset = load_dataset("open-r1/Mixture-of-Thoughts", "all", split="all")

    dataset = dataset.select(range(args.start, args.end))
    dataset = dataset.shuffle(seed=42)

    # System message that will be prepended to all conversations
    system_message = {
        "role": "system",
        "content": "You are a helpful, respectful and honest assistant.",
    }

    def format_conversation_sharegpt(row, dataset_column="conversations"):
        messages = [system_message]
        current_role = None
        for message in row[dataset_column]:
            if message["from"] == "human":
                messages.append({"role": "user", "content": message["value"]})
            elif message["from"] == "gpt":
                messages.append({"role": "assistant", "content": message["value"]})
            else:
                raise ValueError(f"Unknown role: {message['from']}")

            if current_role is None:
                current_role = messages[-1]["role"]
            else:
                assert (
                    current_role != messages[-1]["role"]
                ), f"Conversation has incorrect role order"
                current_role = messages[-1]["role"]
        return {"messages": messages}

    def format_conversation_ultrachat(row, dataset_column="messages"):
        messages = [system_message]
        for message in row[dataset_column]:
            messages.append(message)
        return {"messages": messages}

    if args.dataset == "sharegpt":
        dataset = dataset.map(format_conversation_sharegpt)
    elif args.dataset == "ultrachat":
        dataset = dataset.map(format_conversation_ultrachat)
    elif args.dataset == "mixture_of_thoughts":
        pass  # no need to format

    # ------------------------ 2. Tokenizer ------------------------
    # This step tokenizes the conversation and creates the loss mask
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Special token sequences used to identify different parts of the conversation
    # For Llama models
    # assistant_header = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    # user_header = "<|eot_id|><|start_header_id|>user<|end_header_id|>"
    # For Qwen models
    # assistant_header = "<|im_start|>assistant\n"
    # user_header = "<|im_start|>user\n"
    # for Llama-4-Scout-17B-16E-Instruct
    assistant_header = "<|header_start|>assistant<|header_end|>\n\n"
    user_header = "<|header_start|>user<|header_end|>"

    def tokenize_conversation(row, tokenizer, col="messages"):
        formatted_conversation = tokenizer.apply_chat_template(
            row[col], tokenize=False, add_generation_prompt=False
        )
        encoding = tokenizer(
            formatted_conversation,
            return_offsets_mapping=True,
            max_length=MAX_TOKEN_LENGTH,
        )
        input_ids = encoding.input_ids
        offsets = encoding.offset_mapping
        loss_mask = torch.zeros(len(input_ids), dtype=torch.long)

        # Find spans of assistant responses using regex
        assistant_pattern = (
            re.escape(assistant_header) + r"(.*?)(?=" + re.escape(user_header) + "|$)"
        )
        for match in re.finditer(assistant_pattern, formatted_conversation, re.DOTALL):
            # Assistant response text span (excluding assistant_header itself)
            assistant_start_char = match.start(1)
            assistant_end_char = match.end(1)

            # Mark tokens overlapping with assistant response
            for idx, (token_start, token_end) in enumerate(offsets):
                # Token is part of the assistant response span
                if token_end <= assistant_start_char:
                    continue  # token before assistant text
                if token_start > assistant_end_char:
                    continue  # token after assistant text
                loss_mask[idx] = 1
        return {
            "conversation_str": formatted_conversation,
            "input_ids": input_ids,
            "loss_mask": loss_mask,
        }

    dataset = dataset.map(tokenize_conversation, fn_kwargs={"tokenizer": tokenizer})
    dataset = dataset.remove_columns(
        [
            col
            for col in dataset.column_names
            if col not in ["input_ids", "loss_mask", "conversation_str"]
        ]
    )
    dataset.set_format(type="torch")
    # 上面都是 BaldEagle 里面的。 下面是自己写的 sglang engine。
    # python3 data_gen.py --start 0 --end 2 --index 1 --gpu_index 0 --outdir /root/.cache/hidden_states_dump --model_name meta-llama/Llama-4-Scout-17B-16E-Instruct --dataset sharegpt
    # ------------------------ 3. Compute hidden states ------------------------
    import sglang as sgl

    # Batching is crucial for performance. This will make the producer much faster.
    producer_batch_size = 16
    chunk_size = 100

    async def producer(data_queue, llm_engine, sampling_params, num_consumers):
        batch_input_ids = []
        batch_rows = []

        for idx, row in tqdm(enumerate(dataset), total=len(dataset)):
            batch_input_ids.append(row["input_ids"].tolist())
            batch_rows.append(row)

            if len(batch_input_ids) < producer_batch_size and idx < len(dataset) - 1:
                continue

            # Process the batch asynchronously
            outputs = await llm_engine.async_generate(
                input_ids=batch_input_ids,
                sampling_params=sampling_params,
                return_hidden_states=True,
            )

            # Process outputs and put them in the queue
            for i in range(len(outputs)):
                output_row = outputs[i]
                # print("output_row: ", output_row)
                input_row = batch_rows[i]
                hs_all = output_row["meta_info"]["hidden_states"][0]

                if not hs_all:
                    continue

                hidden_dim = 5120
                tgt_hs_list = []
                hs_list = [[], [], []]

                for token_hiddens in hs_all:
                    token_hiddens = torch.tensor(token_hiddens, dtype=torch.bfloat16)
                    tgt_hs_list.append(token_hiddens[:hidden_dim])
                    remaining = token_hiddens[hidden_dim:]
                    for j in range(3):
                        start_idx = j * hidden_dim
                        end_idx = (j + 1) * hidden_dim
                        hs_list[j].append(remaining[start_idx:end_idx])

                tgt_hs = torch.stack(tgt_hs_list).cpu()
                hs = torch.stack([torch.stack(layer) for layer in hs_list]).cpu() # (3, S, D)

                await data_queue.put(
                    {
                        "input_ids": input_row["input_ids"],
                        "loss_mask": input_row["loss_mask"],
                        "hidden_state": hs,
                        "target_hidden_states": tgt_hs,
                    }
                )

            # Clear batch
            batch_input_ids.clear()
            batch_rows.clear()

        for _ in range(num_consumers):
            await data_queue.put(None)  # Signal end of production
        print("Producer finished.")
        llm_engine.shutdown()
        print("LLM shutdown")

    def save_chunk_sync(buf, out_dir, c_idx):
        """Synchronous function to save a chunk of data to disk."""
        if not buf:
            return
        print(f"Saving chunk {c_idx} with {len(buf)} records...")
        records = [
            {
                "input_ids": item["input_ids"].tolist(),
                "loss_mask": item["loss_mask"].tolist(),
                "hidden_state": item["hidden_state"].float().numpy().tolist(),
                "target_hidden_states": item["target_hidden_states"]
                .float()
                .numpy()
                .tolist(),
            }
            for item in buf
        ]
        ds = Dataset.from_dict({k: [r[k] for r in records] for k in records[0]})
        ds.save_to_disk(f"{out_dir}/chunk_{c_idx}")
        print(f"Finished saving chunk {c_idx}.")

    async def consumer(worker_id, data_queue, outdir):
        buffer = []
        chunk_idx = 0

        while True:
            item = await data_queue.get()
            if item is None:
                break
            buffer.append(item)

            if len(buffer) >= chunk_size:
                # Run the synchronous, blocking IO in a separate thread
                file_chunk_idx = f"{worker_id}_{chunk_idx}"
                await asyncio.to_thread(save_chunk_sync, buffer, outdir, file_chunk_idx)
                buffer.clear()
                chunk_idx += 1

        # Save any remaining items in the buffer
        if buffer:
            file_chunk_idx = f"{worker_id}_{chunk_idx}"
            await asyncio.to_thread(save_chunk_sync, buffer, outdir, file_chunk_idx)
        print(f"Consumer {worker_id} finished.")

    async def async_main():
        llm = sgl.Engine(
            model_path=args.model_name,
            skip_tokenizer_init=True,
            enable_return_hidden_states=True,
            tp_size=8,
            context_length=65536,
            disable_cuda_graph=True,
            disable_radix_cache=True,
        )
        sampling_params = {
            "temperature": 0,
            "max_new_tokens": 0,
        }

        outdir = f"{args.outdir}/{args.index}"
        os.makedirs(outdir, exist_ok=True)
        data_queue = asyncio.Queue(maxsize=200)
        num_consumers = args.num_consumers

        # Run producer and consumer concurrently
        producer_task = asyncio.create_task(
            producer(data_queue, llm, sampling_params, num_consumers)
        )
        consumer_tasks = [
            asyncio.create_task(consumer(i, data_queue, outdir))
            for i in range(num_consumers)
        ]

        await producer_task
        await asyncio.gather(*consumer_tasks)

        print(f"✅ Done! Data has been written to {outdir}")

    asyncio.run(async_main())


if __name__ == "__main__":
    # uvloop can provide better performance if installed
    # try:
    #     import uvloop
    #     uvloop.install()
    # except ImportError:
    #     pass
    main()