"""
download the fineweb pre-training corpus

```
python scripts/data_process/fineweb.py --num_samples=10000000 --min_length_for_memory=2048 --validation_size=3000
```
"""

from typing import Any, Dict, List

from absl import app, flags
from datasets import load_dataset
from transformers import AutoTokenizer

FLAGS = flags.FLAGS

def set_args():
    flags.DEFINE_integer(
        "num_samples",
        default=10_000_000,
        help="number of samples to sample from the FineWeb.",
    )
    flags.DEFINE_integer(
        "min_length_for_memory",
        default=2048,
        help="minimum length for pre-training text subset with memory.",
    )
    flags.DEFINE_integer(
        "validation_size",
        default=3_000,
        help="number of samples for validation set.",
    )




def main(argv):
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    num_samples = FLAGS.num_samples
    dataset = load_dataset(
        "HuggingFaceFW/fineweb",
        name="sample-10BT",
        split="train"
    )

    total_samples = len(dataset)
    print("total samples num: ", total_samples)

    # Extract the last 90,000 samples
    last_half_samples = dataset.select(range(0, total_samples))

    def tokenize_texts(examples: Dict[str, List[Any]]):
        token_counts = tokenizer(examples["text"], add_special_tokens= False)["input_ids"]
        examples["num_tokens"] = [len(x) for x in token_counts]
        return examples

    dataset_with_token_num = last_half_samples.map(tokenize_texts, batched=True, num_proc=192)

    # def filter_fn(example: Dict[str, Any]):
    #     return example["num_tokens"] > FLAGS.min_length_for_memory
    # filtered_dataset = dataset_with_token_num.filter(filter_fn)
    def filter_fn(examples: Dict[str, List[Any]]):
        token_counts = examples["num_tokens"]
        return [x > FLAGS.min_length_for_memory for x in token_counts]
    filtered_dataset = dataset_with_token_num.filter(filter_fn, batched=True, num_proc=192)
    filtered_dataset = filtered_dataset.remove_columns("num_tokens")

    text_mem = filtered_dataset.select(range(0, len(filtered_dataset) // 2))
    text_inst = filtered_dataset.select(range(len(filtered_dataset) // 2, len(filtered_dataset)))

    random_seed = 42
    text = dataset.shuffle(seed=random_seed).select(range(0, num_samples))


    text_mem = text_mem.train_test_split(test_size=FLAGS.validation_size)
    text_inst = text_inst.train_test_split(test_size=FLAGS.validation_size)
    text = text.train_test_split(test_size=FLAGS.validation_size)

    # print("text:", len(text), "textmem:", len(text_mem), "text:", len(text_inst),)
    print("text:", text, "textmem:", text_mem, "text inst:", text_inst,)
    shards = {'train': 128, 'test': 4}
    text.save_to_disk("dataset_cache/processed/fineweb/text", num_shards=shards, num_proc=128)
    text_mem.save_to_disk("dataset_cache/processed/fineweb/text_mem", num_shards=shards, num_proc=128)
    text_inst.save_to_disk("dataset_cache/processed/fineweb/text_inst", num_shards=shards, num_proc=128)

if __name__ == "__main__":
    set_args()
    app.run(main)

