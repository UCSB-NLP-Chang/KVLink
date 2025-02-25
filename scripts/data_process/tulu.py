"""
Generate the SFT subsets without memory

```
python scripts/data_process/tulu.py --max_length=4096 --validation_size=2000
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

# # Specify the column name you're interested in
# column_name = 'source'  # Replace with your column name

# # Get unique values
# unique_values = dataset.unique(column_name)

# # Print each unique value
# for value in unique_values:
#     print(value)

def main(argv):
    shards = {'train': 128, 'test': 4}
    dataset = load_dataset("allenai/tulu-3-sft-mixture", split="train")

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    max_length = FLAGS.max_length

    def sft_filter(sample):

        if "oasst1_converted" in sample['source'] or "tulu_v3.9_aya_100k" in sample["source"] or "tulu_v3.9_wildchat_100k" in sample["source"]:
            return False

        # Extract "Assistant" responses and mask "User" queries
        system = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou're an assistant who answer the question with the knowledge provided in the prompt<|eot_id|>"
        system_tokenized = tokenizer(system, add_special_tokens=False)
        system_input_ids = system_tokenized.input_ids

        input_ids_list = system_input_ids
        labels = [-100] * len(system_input_ids)

        msgs = sample['messages']

        if len(msgs) < 2:
            return False

        for i in range(len(msgs)):
            if msgs[i]["role"] == "user":

                t = "<|start_header_id|>user<|end_header_id|>\n\n" + msgs[i]["content"]  + "<|eot_id|>" 

                tokenized = tokenizer(t, add_special_tokens=False)
                input_ids = tokenized.input_ids

                if len(labels) + len(input_ids) >= max_length: 
                    break

                labels.extend([-100] * len(input_ids))
                input_ids_list += input_ids

            elif msgs[i]["role"] == "assistant":
                t = "<|start_header_id|>assistant<|end_header_id|>\n\n" + msgs[i]["content"]
                tokenized = tokenizer(t, add_special_tokens=False)

                input_ids = tokenized.input_ids
                if len(labels) + len(input_ids) > max_length - 1: 
                    input_ids = input_ids[:max_length - 1 - len(labels)]

                input_ids += [128009]

                labels.extend(input_ids)

                input_ids_list += input_ids

        if len(labels) > max_length:
            return False

        labels = torch.tensor(labels)
        all_zeros = (labels == -100).all()

        if all_zeros.item():
            return False
        else:
            return True

    sft = dataset.filter(sft_filter, num_proc=96)
    sft = sft.train_test_split(test_size=FLAGS.validation_size)

    sft.save_to_disk("dataset_cache/processed/tulu/sft", num_shards=shards, num_proc=128)
    print("sft:", sft)


if __name__ == "__main__":
    set_args()
    app.run(main)
