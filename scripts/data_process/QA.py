"""
Generate the block_qa subsets with and without memory

```
python scripts/data_process/block_qa.py --max_length=4096 --validation_size=2000
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
        help="Max token length for 2Wiki",
    )
    flags.DEFINE_integer(
        "validation_size",
        default=2_000,
        help="number of samples for validation set.",
    )

def main(argv):
    shards = {'train': 128, 'test': 4}

    dataset = load_dataset('json', data_files="data/raw/block_qa/block_qa.jsonl")['train']

    total_num = len(dataset)
    first_half = dataset.select(range(0, total_num // 2))
    second_half = dataset.select(range(total_num // 2, total_num))

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    max_length = FLAGS.max_length

    def qa_filter(sample):
        # Extract "Assistant" responses and mask "User" queries
        system = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a intelligent AI assistant. Please answer questions based on the user's instruction. Below are some reference documents that may help you in answering the user's question."
        system_input_ids = tokenizer(system, add_special_tokens=False).input_ids
        input_ids = system_input_ids

        if len(sample['documents']) != 10:
            print("Context number not right")
            return False

        for j in range(0,10):
            title = sample['documents'][j]['title']
            text = sample['documents'][j]['text']
            # memory_list.append("<MEM_START>" + f"Document [{j+1}](Title: {title}) {text}" + "\n<MEM_END><MEM_SUM>")
            tem_id = tokenizer(f"Document [{j+1}](Title: {title}) {text}\n", add_special_tokens=False).input_ids

            input_ids += tem_id

        user = "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" + sample['question'] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" +  sample['generated']
        user_id = tokenizer(user, add_special_tokens=False).input_ids
        input_ids += user_id

        if len(input_ids) >= max_length:
            return False

        return True

    qa = first_half.filter(qa_filter, num_proc=96)
    qa = qa.train_test_split(test_size=FLAGS.validation_size)

    qa.save_to_disk("dataset_cache/processed/block_qa/qa", num_shards=shards, num_proc=128)
    print("qa:", qa)

    def qamem_filter(sample):
        # Extract "Assistant" responses and mask "User" queries
        system = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a intelligent AI assistant. Please answer questions based on the user's instruction. Below are some reference documents that may help you in answering the user's question."
        system_input_ids = tokenizer(system, add_special_tokens=False).input_ids
        input_ids = system_input_ids

        if len(sample['documents']) != 10:
            print("Context number not right")
            return False

        for j in range(0,10):
            title = sample['documents'][j]['title']
            text = sample['documents'][j]['text']
            tem_id = tokenizer("<MEM_START>" + f"Document [{j+1}](Title: {title}) {text}\n<MEM_END><MEM_SUM>", add_special_tokens=False).input_ids

            input_ids += tem_id

        user = "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" + sample['question'] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" +  sample['generated']
        user_id = tokenizer(user, add_special_tokens=False).input_ids
        input_ids += user_id

        if len(input_ids) >= max_length:
            return False

        return True

    qamem = second_half.filter(qamem_filter, num_proc=96)
    qamem = qamem.train_test_split(test_size=FLAGS.validation_size)

    qamem.save_to_disk("dataset_cache/processed/block_qa/qamem", num_shards=shards, num_proc=128)
    print("qamem:", qamem)

if __name__ == "__main__":
    set_args()
    app.run(main)
