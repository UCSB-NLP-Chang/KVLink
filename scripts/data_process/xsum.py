"""
Generate the XSum training set

```
python scripts/data_process/sum.py --max_length=4096 --validation_size=1000
```
"""
import json

from absl import app, flags
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

FLAGS = flags.FLAGS

def set_args():
    flags.DEFINE_integer(
        "max_length",
        default=4096,
        help="Max token length for sum",
    )
    flags.DEFINE_integer(
        "validation_size",
        default=2_000,
        help="number of samples for validation set.",
    )

def main(argv):
    shards = {'train': 128, 'test': 4}

    dataset = load_dataset("EdinburghNLP/xsum", split="train")

    total_num = len(dataset)

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    max_length = FLAGS.max_length

    def xsum_filter(sample):
        # Extract "Assistant" responses and mask "User" queries
        system = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a intelligent AI assistant. Please summarize the text based on the information given."
        sys_id = tokenizer(system, add_special_tokens=False).input_ids

        text = sample['document']
        text_id = tokenizer(text, add_special_tokens=False).input_ids

        if len(text_id) < 1000:
            return False

        user = "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" + sample['document'] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" +  sample['summary']

        user_id = tokenizer(user, add_special_tokens=False).input_ids
        input_ids = sys_id + user_id

        if len(input_ids) >= 3000:
            return False

        return True

    xsum = dataset.filter(xsum_filter, num_proc=96)
    xsum = xsum.train_test_split(test_size=FLAGS.validation_size)

    xsum.save_to_disk("dataset_cache/processed/xsum/xsum", num_shards=shards, num_proc=128)
    print("XSum:", xsum)

if __name__ == "__main__":
    set_args()
    app.run(main)