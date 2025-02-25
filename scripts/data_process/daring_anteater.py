"""
Generate the SFT subsets with and without memory

```
python scripts/data_process/daring_anteater.py --max_length=4096 --validation_size=2000
```
"""
import torch
from absl import app, flags
from datasets import load_dataset
from transformers import AutoTokenizer

FLAGS = flags.FLAGS

def set_args():
    flags.DEFINE_integer(
        "max_length",
        default=4096,
        help="Max token length for daring anteater",
    )
    flags.DEFINE_integer(
        "validation_size",
        default=2_000,
        help="number of samples for validation set.",
    )


def main(argv):
    shards = {'train': 128, 'test': 4}
    dataset = load_dataset("nvidia/Daring-Anteater", split="train")

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    max_length = FLAGS.max_length

    def process_sft(conversation):
        # Extract "Assistant" responses and mask "User" queries
        system = "[<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou're an assistant who answer the question with the knowledge provided in the prompt<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        system_tokenized = tokenizer(system, add_special_tokens=False)
        system_input_ids = system_tokenized.input_ids

        input_ids_list = system_input_ids
        labels = [-100] * len(system_input_ids)

        for i in range(len(conversation)):
            if conversation[i]["from"] == "User":
                if i==0:
                    t = conversation[i]["value"] + "<|eot_id|>"
                else:
                    t = "<|start_header_id|>user<|end_header_id|>\n\n" + conversation[i]["value"]  + "<|eot_id|>" 

                tokenized = tokenizer(t)

                input_ids = tokenized.input_ids[1:]
                if len(labels) + len(input_ids) >= max_length: 
                    break

                labels.extend([-100] * len(input_ids))
                input_ids_list += input_ids

            elif conversation[i]["from"] == "Assistant":
                t = "<|start_header_id|>assistant<|end_header_id|>\n\n" + conversation[i]["value"]
                tokenized = tokenizer(t)

                input_ids = tokenized.input_ids[1:]
                if len(labels) + len(input_ids) > max_length - 1: 
                    input_ids = input_ids[:max_length - 1 - len(labels)]

                input_ids += [128009]

                labels.extend(input_ids)

                input_ids_list += input_ids
        return {
            'input_ids': input_ids_list,
            'labels': labels
        }

    def sft_filter(sample):
        labels = torch.tensor(process_sft(sample['conversations'])['labels'])
        all_zeros = (labels == -100).all()
        if all_zeros.item():
            return False
        else:
            return True

    sft = dataset.filter(sft_filter, num_proc=96)
    sft = sft.train_test_split(test_size=FLAGS.validation_size)

    sft.save_to_disk("dataset_cache/processed/daringanteater/sft", num_shards=shards, num_proc=128)
    print("sft:", sft)

    def process_sftmem(conversation):
        sys = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou're an assistant who answer the question with the knowledge provided in the prompt<|eot_id|>"
        sys_tokens = tokenizer(sys, add_special_tokens= False, return_tensors= "pt")['input_ids']
        sys_len = sys_tokens.size(1)

        if len(conversation) % 2 != 0:
            if conversation[0]["from"] == "Assistant":
                conversation = conversation[1:]
            elif conversation[-1]["from"] == "User":
                conversation = conversation[:-1]
            else:
                conversation = conversation[:-1]
        memory_ids = []
        memory_positions = []
        current_position = sys_len
        for idx in range(0, len(conversation) - 2, 2):
            if (
                conversation[idx]["from"] == "User" and
                conversation[idx + 1]["from"] == "Assistant"
            ):
                text = "<|start_header_id|>user<|end_header_id|>\n\n" + conversation[idx]["value"] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" + conversation[idx + 1]["value"] + "<|eot_id|>"
                memory_tokens = tokenizer(text, add_special_tokens= False, return_tensors= "pt")['input_ids']
                memory_tokens = torch.cat([torch.tensor([[128256]]).to(memory_tokens.device), memory_tokens, torch.tensor([[128257]]).to(memory_tokens.device)], dim = 1)
                memory_ids.append(memory_tokens[0])

                mem_len = memory_tokens.size(1)
                memory_positions.append(torch.arange(current_position, current_position + mem_len))
                current_position += mem_len

        last_q = "<|start_header_id|>user<|end_header_id|>\n\n" + conversation[len(conversation) - 2]["value"] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        remaining_ids = tokenizer(last_q, add_special_tokens= False, return_tensors= "pt")['input_ids']
        remaining_ids = torch.cat([torch.tensor([[128258]]), remaining_ids], dim = 1)
        labels = torch.tensor([[-100] * remaining_ids.size(1)])

        last_a = conversation[len(conversation) - 1]["value"] + "<|eot_id|>"
        answer_tokens = tokenizer(last_a, add_special_tokens= False, return_tensors= "pt")['input_ids']
        remaining_ids = torch.cat([remaining_ids, answer_tokens], dim = 1)
        labels = torch.cat([labels, answer_tokens], dim = 1)


        if(current_position + labels.size(1) >= max_length):
            return  False
        else:
            return True

    def filter_sftmem(sample):
        if(len(sample['conversations']) <= 2 or len(sample['conversations']) % 2 == 1):
            return False
        return process_sftmem(sample['conversations'])

    sft_mem = dataset.filter(filter_sftmem, num_proc=96)
    sft_mem = sft_mem.train_test_split(test_size=FLAGS.validation_size)

    sft_mem.save_to_disk("dataset_cache/processed/daringanteater/sft_mem", num_shards=shards, num_proc=128)
    print("sft_mem:", sft_mem)

if __name__ == "__main__":
    set_args()
    app.run(main)
