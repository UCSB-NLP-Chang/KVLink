"""
Evaluate the performance of LLMs fine-tuned via TorchTitan on hqa dataset.

1. Convert the Titan checkpoint from DCP to torch:
```
python -m torch.distributed.checkpoint.format_utils dcp_to_torch \
    torchtitan/outputs/checkpoint/step-1000 checkpoint.pt
```

2. Run evaluation:
```
python scripts/evaluation/hqa_eval.py \
    --ckpt_path checkpoint.pt \
    --batch_size 10 \
    --reencode_num 5 \
    --attn_type "blocked" \
```
"""
import argparse
import datetime
import json
import os
import string
from typing import Dict, List

import datasets
import numpy as np
import regex
import torch
from safetensors import safe_open
from torch.utils.data import DataLoader
from torchtune.models.convert_weights import tune_to_hf
from tqdm.auto import tqdm as auto_tqdm
from transformers import AutoTokenizer, GenerationConfig, LlamaForCausalLM, AutoModelForCausalLM

from src.common import move_to_target_device
from src.data.titan_preprocessor import LLaMA32Tokenizer, make_segment_mask

parser = argparse.ArgumentParser(description="Run script with specified ckpt.")
parser.add_argument(
    "--ckpt_path",
    type=str,
    default=None,
    help="The path to the `checkpoint.pt` file.",
)
parser.add_argument("--batch_size", type=int, default=1, help="Batch size of the evaluation.")
parser.add_argument(
    "--attn_type",
    type=str,
    required=True,
    help="attention types.",
    choices=["standard", "blocked"],
)
parser.add_argument("--reencode_num", type=int, default=5, help="Number of the link tokens.")
parser.add_argument("--hf", type=bool, default=False, help="Use HuggingFace checkpoints.")

args = parser.parse_args()

def load_model_weights(ckpt_path: str):
    safe_tensor_file = os.path.join(ckpt_path, "model.safetensors")
    if os.path.exists(safe_tensor_file):
        state_dict = {}
        with safe_open(safe_tensor_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
        # state_dict["output.weight"] = state_dict["tok_embeddings.weight"]
        return state_dict

    state_dict = torch.load(ckpt_path, weights_only=False)

    state_dict = state_dict["model"]
    state_dict["output.weight"] = state_dict["tok_embeddings.weight"]

    converted_state_dict = tune_to_hf(
        state_dict=state_dict,
        num_heads=32,
        num_kv_heads=8,
        dim=2048,
    )
    return converted_state_dict

def normalize_answer(s: str) -> str:
    """Normalization from the SQuAD evaluation script.

    See https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
    """

    def remove_articles(text):
        return regex.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def best_subspan_em(prediction: str, ground_truths: List[str]) -> float:
    normalized_prediction = normalize_answer(prediction)

    for ground_truth in ground_truths:
        normalized_ground_truth = normalize_answer(ground_truth)
        if normalized_ground_truth.lower() in normalized_prediction.lower():
            return 1.0
    return 0.0

def preprocess_fn(example: Dict[str, str], tokenizer: LLaMA32Tokenizer, reencode_num: int, mem_start: int, mem_end: int, special_token_start:int):
    system = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a intelligent AI assistant. Please answer questions based on the user's instruction. "
        "Below are some reference documents that may help you in answering the user's question.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    )
    question = example["question"]
    memory_list = []

    for j in range(len(example['context']['title'])):
        title = example['context']['title'][j]
        text = ''.join(example['context']['sentences'][j])
        memory_list.append(f"Document [{j+1}](Title: {title}) {text}\n")

    qa_system_input_ids = tokenizer(system, add_special_tokens = False)["input_ids"]
    input_ids = qa_system_input_ids[:] + [mem_start]
    segment_ids = [0] * len(input_ids)

    for mem_id, st in enumerate(memory_list):
        tem_id = tokenizer(st, add_special_tokens = False)["input_ids"]
        segment_ids = segment_ids + [mem_id + 1] * len(tem_id) + [0] * reencode_num

        for sub_idx in range(reencode_num):
            tem_id = tem_id + [special_token_start + reencode_num * mem_id + sub_idx]
        input_ids = input_ids + tem_id

    new_prompt = question
    prompt_id = tokenizer(new_prompt, add_special_tokens = False)["input_ids"]
    input_ids = input_ids + [mem_end] + prompt_id
    segment_ids = segment_ids + [0] + [0] * len(prompt_id)
    return {
        "input_ids": input_ids,
        "segment_ids": segment_ids,
    }

class DataCollatorForGeneration():
    def __init__(self, pad_id: int):
        self.pad_id = pad_id
        pass
    def __call__(self, batch):
        input_ids = []
        segment_ids = []
        attention_mask = []
        length_list = [len(x['input_ids']) for x in batch]

        max_length = max(length_list)

        for item in batch:
            seq_length = len(item['input_ids'])

            residual = max_length - seq_length
            # padded_input_ids = [self.pad_id] * residual + item['input_ids']
            # curr_attention_mask = [0] * residual + [1] * seq_length
            padded_input_ids = item['input_ids'] + [self.pad_id] * residual
            curr_attention_mask = [1] * seq_length + [0] * residual
            input_ids.append(padded_input_ids)
            attention_mask.append(curr_attention_mask)
            segment_ids.append(item["segment_ids"] + [-1] * residual)

        return {
            "input_ids": torch.LongTensor(input_ids),
            "segment_ids": torch.LongTensor(segment_ids),
            "attention_mask": torch.LongTensor(attention_mask),
        }


def main():
    ckpt_path = args.ckpt_path
    reencode_num: int  = args.reencode_num
    batch_size: int = args.batch_size
    device = torch.device("cuda")
    hf: bool = args.hf

    mem_start = 128254
    mem_end = 128255
    special_token_start = 128011

    dataset = datasets.load_dataset("hotpotqa/hotpot_qa", 'distractor', split='validation')
    print(dataset)
    all_answers = dataset["answer"]
    print(all_answers[:10])

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    tokenizer.pad_token_id = 128004
    tokenizer.pad_token = "<|finetune_right_pad_id|>"

    model = LlamaForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct",
        torch_dtype=torch.bfloat16,
        # torch_dtype=torch.float32,
    )
    # model.load_state_dict(state_dict, strict=True)

    if args.ckpt_path is None:
        print("Will NOT load fine-tuned models!")
    elif hf:
        model = AutoModelForCausalLM.from_pretrained(args.ckpt_path, torch_dtype=torch.bfloat16)
    else:
        state_dict = load_model_weights(ckpt_path)
        model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    exist_columns = dataset.column_names
    dataset = dataset.map(
        preprocess_fn,
        batched=False,
        num_proc=16,
        remove_columns=exist_columns,
        fn_kwargs=dict(
            tokenizer=tokenizer,
            reencode_num=reencode_num,
            mem_start=mem_start,
            mem_end=mem_end,
            special_token_start=special_token_start
        ),
    )

    total_num = len(dataset)
    dataset = dataset.select(np.arange(total_num))
    correct_num = 0
    res_list = []

    collate_fn = DataCollatorForGeneration(pad_id=tokenizer.pad_token_id)
    eval_dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    prog_bar = auto_tqdm(range(len(eval_dataloader)))

    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    print(eot_id)
    generation_cfg = GenerationConfig(
        do_sample=False,
        num_beams=1,
        max_new_tokens=200,
        stop_strings=["<|end_of_text|>", "<|eot_id|>"],
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=eot_id,
    )
    generation_prompt = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

    generation_token_ids = tokenizer(generation_prompt, add_special_tokens=False)["input_ids"]
    print(generation_token_ids)
    generation_token_ids = torch.LongTensor(generation_token_ids)
    generation_token_ids: torch.LongTensor = move_to_target_device(generation_token_ids, device)

    for batch_id, batch in enumerate(eval_dataloader):
        curr_batch_size = batch['input_ids'].size(0)
        batch_answers = all_answers[batch_id * batch_size : batch_id * batch_size + curr_batch_size]
        segment_ids = batch["segment_ids"]
        attention_mask = make_segment_mask(
            source_segments=segment_ids,
            target_segments=segment_ids,
            add_causal_lm_mask=True,
        )
        attention_mask_4d = attention_mask.unsqueeze(1)
        input_ids = batch["input_ids"]
        attention_mask_for_pad = batch["attention_mask"]

        with torch.no_grad():
            input_ids = move_to_target_device(input_ids, device)
            attention_mask_4d = move_to_target_device(attention_mask_4d, device)
            attention_mask_for_pad = move_to_target_device(attention_mask_for_pad, device)

            if args.attn_type == "blocked":
                prefilling_outputs = model(input_ids=input_ids, attention_mask=attention_mask_4d)
            elif args.attn_type == "standard":
                prefilling_outputs = model(input_ids=input_ids, attention_mask=attention_mask_for_pad)
            else:
                raise ValueError()
            past_key_values = prefilling_outputs.past_key_values


            generation_prefix = generation_token_ids.repeat(curr_batch_size, 1)
            generation_input_ids = torch.cat([input_ids, generation_prefix], axis=1)
            attention_mask_for_pad = torch.cat([attention_mask_for_pad, torch.ones_like(generation_prefix)], axis=1)
            outputs = model.generate(
                input_ids=generation_input_ids,
                attention_mask=attention_mask_for_pad,
                use_cache=True,
                generation_config=generation_cfg,
                past_key_values=past_key_values,
                tokenizer=tokenizer,
            )
        generated_seqs = [tokenizer.decode(
                outputs[i, input_ids.size(1):].tolist(),
            )
            for i in range(input_ids.size(0))
        ]

        responses = [
            generated_seq.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip().split("<|eot_id|>")[0]
            for generated_seq in generated_seqs
        ]
        for idx, x in enumerate(responses):
            print(x)
            print("Ground-truth: ", batch_answers[idx])
            print("------\n")

        scores = [best_subspan_em(responses[idx], [batch_answers[idx]]) for idx in range(curr_batch_size)]
        for idx, score in enumerate(scores):
            correct_num = correct_num + int(score)
            res_list.append(
                {
                    # "question": question,
                    "response": responses[idx],
                    "gold_answer": batch_answers[idx],
                    "score": scores[idx],
                }
            )
        print("Correct progress", correct_num)
        prog_bar.update(1)

    accuracy = correct_num / total_num
    print(accuracy)

    current_time = datetime.datetime.now()
    time_str = current_time.strftime("%Y%m%d-%H%M%S")

    file_name = f"result/hqa_{accuracy}_{time_str}.jsonl"
    if not os.path.exists(os.path.dirname(file_name)):
        os.makedirs(os.path.dirname(file_name))

    with open(file_name, "w", encoding="utf-8") as f:
        for entry in res_list:
            json_line = json.dumps(entry)
            f.write(json_line + "\n")

    print(f"Dumped at {file_name}")

if __name__ == "__main__":
    main()
