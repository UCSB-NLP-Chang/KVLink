"""
retrieve the relevant documents and use GPT4 to generate answer for each question

```
python scripts/data_process/gpt_answer.py
```
"""
import json
import os
import random
from typing import Any, Dict, List

import pandas as pd
import torch
from tqdm import tqdm
from dotenv import load_dotenv
from openai import AzureOpenAI
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

load_dotenv()
api_key = os.getenv('api_key')
api_version = os.getenv('api_version')
azure_endpoint = os.getenv('azure_endpoint')

@torch.no_grad()
def compute_embeddings(sentences: List[str], model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    for k in inputs:
        inputs[k] = inputs[k].to(device=model.device, dtype=torch.int64)
    outputs = model(**inputs)
    embeddings = mean_pooling(token_embeddings=outputs[0], mask=inputs["attention_mask"])
    return embeddings

def mean_pooling(token_embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    token_embeddings = token_embeddings.masked_fill_(~mask[..., None].bool(), 0.0)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

def process_tqa_instance(ins: Dict[str, Any], model:PreTrainedModel, retrieval_tokenizer:PreTrainedTokenizer):
    documents = [{"title":i['title'], "text":i['text'], "score":0.0} for i in ins["ctxs"]]
    embeddings = compute_embeddings(
        sentences=[ins['question']] + [i['text'] for i in documents], model=model, tokenizer=retrieval_tokenizer
    )
    q_emb = embeddings[0].clone().unsqueeze(dim=0)
    scores = torch.matmul(input=q_emb, other=embeddings[1:].T).squeeze(dim=0)
    values, indices = torch.sort(input=scores, descending=True)
    values, indices = values.tolist(), indices.tolist()

    for idx, score in zip(indices, values):
        documents[idx]['score'] = score
    documents.sort(key=lambda i: i['score'], reverse=True)

    return {
        "question":ins['question'],
        "answers":ins['answers'],
        "generated":"",
        "documents":documents[:10]
    }

def process_tqa(input_file: str, model:PreTrainedModel, retrieval_tokenizer:PreTrainedTokenizer):
    with open(input_file, "r", encoding="utf-8") as f:
        tqa_instances: List[Dict[str, Any]] = json.load(f)

    dataset = []
    for i in tqdm(range(0, len(tqa_instances)), desc="Process TQA: ", total=len(tqa_instances)):
        ins = process_tqa_instance(ins=tqa_instances[i], model=model, retrieval_tokenizer=retrieval_tokenizer)
        dataset.append(ins)

    return dataset

def process_2wiki_instance(ins: Dict[str, Any], model:PreTrainedModel, retrieval_tokenizer:PreTrainedTokenizer):
    if isinstance(ins['context'], str):
        ins['context'] = json.loads(ins['context'])

    documents = [{"title":i[0], "text":''.join(i[1]), "score":0.0} for i in ins["context"]]
    embeddings = compute_embeddings(
        sentences=[ins['question']] + [i['text'] for i in documents], model=model, tokenizer=retrieval_tokenizer
    )
    q_emb = embeddings[0].clone().unsqueeze(dim=0)
    scores = torch.matmul(input=q_emb, other=embeddings[1:].T).squeeze(dim=0)
    values, indices = torch.sort(input=scores, descending=True)
    values, indices = values.tolist(), indices.tolist()

    for idx, score in zip(indices, values):
        documents[idx]['score'] = score
    documents.sort(key=lambda i: i['score'], reverse=True)

    return {
        "question":ins['question'],
        "answers":[ins['answer']],
        "generated":"",
        "documents":documents[:10]
    }

def process_2wiki(input_file: str, model:PreTrainedModel, retrieval_tokenizer:PreTrainedTokenizer):
    df = pd.read_parquet(path=input_file)
    wiki_instances = df.to_dict(orient="records")

    dataset = []
    for i in tqdm(range(0, len(wiki_instances)), desc="Process 2wiki: ", total=len(wiki_instances)):
        ins = process_2wiki_instance(ins=wiki_instances[i], model=model, retrieval_tokenizer=retrieval_tokenizer)
        dataset.append(ins)

    return dataset

def completion_with_backoff_mcopenai(**kwargs):
    client = AzureOpenAI(
        # https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#rest-api-versioning
        api_version=api_version,
        # https://learn.microsoft.com/en-us/azure/cognitive-services/openai/how-to/create-resource?pivots=web-portal#create-a-resource
        azure_endpoint=azure_endpoint,
        api_key=api_key,
    )
    result = client.chat.completions.create(
        model="gpt-4o-mini",
        **kwargs,
    )
    return result

def generate_answer(dataset: List):

    system_msg = "You are an intelligent AI assistant. Please answer questions based on the user's instructions. Below are some reference documents that may help you in answering the user's question.\n\n"
    for i in tqdm(range(0, len(dataset)), desc="Generating answer: ", total=len(dataset)):
        user_msg = ""
        data = dataset[i]
        for j in range(len(data["documents"])):
            user_msg += f"Document [{j+1}](Title: {data['documents'][j]['title']}) {data['documents'][j]['text']}\n"
        user_msg += data["question"]

        message = [{"role":"system", "content":system_msg},{"role":"user", "content":user_msg}]
        response = completion_with_backoff_mcopenai(messages = message, temperature = 0.7, max_tokens=200).choices[0].message.content

        dataset[i]["generated"] = response

    return dataset


def load_jsonline(fp: str):
    with open(fp, "r", encoding="utf-8") as f:
        return [json.loads(i) for i in f]


def write_jsonline(fp: str, obj: List[Any]):
    with open(fp, "w", encoding="utf-8") as f:
        for i in obj:
            f.write(json.dumps(i, ensure_ascii=False) + "\n")


if __name__ == '__main__':
    random.seed(42)
    tqa_path = "FiD/open_domain_data/TQA/train.json"
    wiki_path = "2WikiMultihopQA/train.parquet"
    model_name = "facebook/contriever-msmarco"
    retrieval_tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)
    model: PreTrainedModel = AutoModel.from_pretrained(
        pretrained_model_name_or_path=model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0"
    )
    tqa_data = process_tqa(tqa_path, model, retrieval_tokenizer)
    wiki_data = process_2wiki(wiki_path, model, retrieval_tokenizer)
    num_samples = 20000
    tqa_data = random.sample(population=tqa_data, k=num_samples)
    wiki_data = random.sample(population=wiki_data, k=num_samples)
    merged_data = tqa_data + wiki_data
    random.shuffle(merged_data)
    generate_answer(merged_data)
    write_jsonline("data/raw/block_qa/block_qa.jsonl",merged_data)
