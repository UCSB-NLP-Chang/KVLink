"""
On `segment_ids`:
    * A tensor (actually a list of list during tokenization) of shape [batch, input_length]
        with values in [0, num_segments].
    * Tokens are only allowed to attend to:
        1. other tokens within the same segment (memory).
        2. segment_ids == 0, for system prompts or content that does not belongs to memory.
    * None represents an all-one tensor, i.e. all positions are in the same segment.
"""
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch
from transformers import PreTrainedTokenizerBase

from src.data.titan_tokenizer import LLaMA32Tokenizer

NEG_INF = -1e15

class SumAttentionPreprocessor():
    '''
    Apply one piece of memory to non-memory use samples to enable batch forward pass for calculating KV.
    '''
    def __init__(
        self,
        tokenizer: Union[LLaMA32Tokenizer, PreTrainedTokenizerBase],
        max_len: int,
        special_token_start: int,
        mem_start: int,
        mem_end: int,
        reencode_num: int,
        min_memory_num: int = 5,
        max_memory_num: int = 40,
        qa_document_num: int = 10,
        min_mem_length: int = 500,
        max_mem_length: int = 1500,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.special_token_start = special_token_start
        self.mem_start = mem_start
        self.mem_end = mem_end
        self.reencode_num = reencode_num
        self.min_memory_num = min_memory_num
        self.max_memory_num = max_memory_num
        self.qa_document_num = qa_document_num
        self.min_memory_length = min_mem_length
        self.max_memory_length = max_mem_length

        if isinstance(tokenizer, LLaMA32Tokenizer):
        # self.bos_token_id = self.tokenizer("<|begin_of_text|>")["input_ids"][0]
            self.bos_token_id = self.tokenizer.bos_id
            self.use_hf_tokenizer = False
        else:
            self.bos_token_id = self.tokenizer.bos_token_id
            self.use_hf_tokenizer = True
        self.prepare_preprocessor()

    def prepare_preprocessor(self,):

        self.sft_system = [
            "You are an AI assistant. Provide helpful, accurate, and clear answers. When uncertain, explain your reasoning or request clarification.",
            "You are an AI assistant. Focus on achieving the user's goal in each interaction. Use concise yet informative explanations.",
            "You are an AI assistant. Speak clearly and stay consistent with prior statements. If you need more information, politely ask for it.",
            "You are an AI assistant. Provide truthful, well-sourced information whenever possible. Acknowledge any limitations and avoid speculation if unsure."
        ]
        self.sft_system_input_id_list = [self.tokenizer("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" + system_prompt + "<|eot_id|>", add_special_tokens=False)["input_ids"] for system_prompt in self.sft_system]

        self.qa_system = [
            "You are an AI assistant. Use the provided documents to answer the user’s question. If the information is insufficient, acknowledge the gap or request clarification.",
            "You are an AI assistant. Always ground your answers in the retrieved documents and do not add unsupported details. If the documents lack sufficient information, indicate that.",
            "You are an AI assistant. Rely solely on the given documents for evidence when answering questions. When necessary, cite or paraphrase the document content accurately.",
            "You are an AI assistant. Base your replies on the retrieved documents, ensuring completeness and correctness. Ask for more details if the documents do not cover the question fully."
        ]
        self.qa_system_input_id_list = [self.tokenizer("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" + system_prompt + "<|eot_id|>", add_special_tokens=False)["input_ids"] for system_prompt in self.qa_system]

        self.summary_system = [
            "You are an AI assistant. Read the provided text and produce a concise summary. Capture the main points without unnecessary detail.",
            "You are an AI assistant. Summarize the essential ideas from the given text. Avoid minor details and focus on critical insights.",
            "You are an AI assistant. Provide a brief, high-level overview of the text. Ensure clarity and coherence, prioritizing key themes.",
            "You are an AI assistant. Summarize the text clearly and logically. Organize the main ideas in a coherent sequence."
        ]
        self.summary_system_input_id_list = [self.tokenizer("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" + system_prompt + "<|eot_id|>", add_special_tokens=False)["input_ids"] for system_prompt in self.summary_system]

        text_inst_system = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou're an assistant who will complete the sentence after the text chunks given below<|eot_id|>"
        text_inst_sys_tokens = self.tokenizer(text_inst_system, add_special_tokens= False)['input_ids']
        self.text_inst_sys_tokens = text_inst_sys_tokens + [self.mem_start]

        user_start_tokens = "<|start_header_id|>user<|end_header_id|>\n\n"
        self.user_start_token_ids = self.tokenizer(
            user_start_tokens, add_special_tokens=False
        )["input_ids"]
        self.eot_token_id = self.tokenizer("<|eot_id|>", add_special_tokens=False)["input_ids"][0]
        print("EOT id: ", self.eot_token_id)
        assistant_start_tokens = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        self.assistant_start_token_ids = self.tokenizer(
            assistant_start_tokens, add_special_tokens=False
        )["input_ids"]

        self.all_memory_sum_tokens = [
            [
                self.special_token_start + idx * self.reencode_num + offset
                for offset in range(self.reencode_num)
            ]
            for idx in range(self.max_memory_num)
        ]

    def process_sftmem(
        self,
        example: Dict[str, str],
    ):
        conversation = example['conversations']
        input_texts = [conversation[i]["value"] for i in range(len(conversation))]
        if self.use_hf_tokenizer:
            all_conversation_texts_ids = self.tokenizer(
                input_texts,
                add_special_tokens=False,
                padding=False,
                truncation=False,
                return_attention_mask=False,
                return_token_type_ids=False,
                return_offsets_mapping=False,
                return_special_tokens_mask=False,
                return_length=False,
            )["input_ids"]
        else:
            all_conversation_texts_ids = [
                self.tokenizer(x, add_special_tokens=False)["input_ids"] for x in input_texts
            ]

        sft_mem_system_input_ids = random.choice(self.sft_system_input_id_list)
        input_ids = sft_mem_system_input_ids + [self.mem_start]
        labels = [-100] * (len(sft_mem_system_input_ids) + 1)
        segment_ids = [0] * (len(sft_mem_system_input_ids) + 1)

        for idx in range(0, len(conversation) - 2, 2):
            if (
                conversation[idx]["from"] == "User" and
                conversation[idx + 1]["from"] == "Assistant"
            ):
                chat_input_ids = (
                    self.user_start_token_ids + all_conversation_texts_ids[idx]
                    + [self.eot_token_id] + self.assistant_start_token_ids
                    + all_conversation_texts_ids[idx + 1] + [self.eot_token_id]
                )
                chat_input_ids = chat_input_ids + self.all_memory_sum_tokens[int(idx / 2)]
                input_ids = input_ids + chat_input_ids

                mem_len = len(chat_input_ids)

                chat_segment_ids = [(idx + 2) // 2] * (mem_len - self.reencode_num) + [0] * self.reencode_num
                segment_ids = segment_ids + chat_segment_ids

        last_q_input_ids = (
            [self.mem_end] + self.user_start_token_ids + all_conversation_texts_ids[-2]
            + [self.eot_token_id] + self.assistant_start_token_ids
            + all_conversation_texts_ids[-1] + [self.eot_token_id]
        )
        last_q_segment_ids = [0] * len(last_q_input_ids)

        input_ids = input_ids + last_q_input_ids
        segment_ids = segment_ids + last_q_segment_ids

        seq_len = len(input_ids)
        ans_len = len(all_conversation_texts_ids[-1]) + 1
        labels = (
            [-100] * (seq_len - ans_len) + all_conversation_texts_ids[-1] + [self.eot_token_id]
        )

        return {
            "input_ids": input_ids,
            "labels": labels,
            "segment_ids": segment_ids
        }

    def process_sft(
        self,
        example: Dict[str, str],
    ):
        conversation = example['conversations']
        input_texts = [conversation[i]["value"] for i in range(len(conversation))]
        if self.use_hf_tokenizer:
            all_conversation_texts_ids = self.tokenizer(
                input_texts,
                add_special_tokens=False,
                padding=False,
                truncation=False,
                return_attention_mask=False,
                return_token_type_ids=False,
                return_offsets_mapping=False,
                return_special_tokens_mask=False,
                return_length=False,
            )["input_ids"]
        else:
            all_conversation_texts_ids = [
                self.tokenizer(x, add_special_tokens=False)["input_ids"] for x in input_texts
            ]
        sft_system_input_ids = random.choice(self.sft_system_input_id_list)
        input_ids = sft_system_input_ids
        labels = [-100] * len(sft_system_input_ids)
        for i in range(len(conversation)):
            if conversation[i]["from"] == "User":
                user_msg_input_ids = (
                    self.user_start_token_ids + all_conversation_texts_ids[i]
                    + [self.eot_token_id]
                )
                if len(labels) + len(user_msg_input_ids) >= self.max_len:
                    break

                labels.extend([-100] * len(user_msg_input_ids))
                input_ids += user_msg_input_ids

            # TODO (KVLinkDeveloper): Currently always stop with EOT if exceed the max length
            # Should revise it so that it is not truncated and append an EOT.
            # Just truncated at the max_len position is enough. Do not have to
            # end with EOT
            elif conversation[i]["from"] == "Assistant":
                assist_msg_input_ids = (
                    self.assistant_start_token_ids + all_conversation_texts_ids[i]
                )
                if len(labels) + len(assist_msg_input_ids) > self.max_len - 1:
                    assist_msg_input_ids = input_ids[:self.max_len - 1 - len(labels)]

                assist_msg_input_ids += [self.eot_token_id]
                labels.extend(assist_msg_input_ids)
                input_ids += assist_msg_input_ids

        # No memory. So, the segment ids are just the same for all positions
        segment_ids = [0] * len(input_ids)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "segment_ids": segment_ids,
        }

    def process_textinst(
        self,
        example: Dict[str, str],
    ):
        sys_len = len(self.text_inst_sys_tokens)

        user = "<|start_header_id|>user<|end_header_id|>\n\nPlease complete the sentence<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        user_tokens = self.tokenizer(user, add_special_tokens= False)['input_ids']
        user_tokens = [self.mem_end] + user_tokens
        user_len = len(user_tokens)

        text = example["text"]
        input_ids = self.tokenizer(text, add_special_tokens= False)['input_ids']

        input_ids = input_ids[:self.max_len - user_len - sys_len]

        mem_len = random.randint(self.min_memory_length, self.max_memory_length)
        mem_num = random.randint(self.min_memory_num, self.max_memory_num)

        breaks = sorted(random.sample(range(1, mem_len), mem_num - 1))
        breaks = [0] + breaks + [mem_len]
        each_mem_len = [breaks[i+1] - breaks[i] for i in range(mem_num)]

        # allocate space for special tokens
        input_len = len(input_ids)
        input_ids = input_ids[:input_len - self.reencode_num * mem_num]

        memory_ids = input_ids[:mem_len]
        remaining_ids = input_ids[mem_len:]

        segment_ids = [0] * len(self.text_inst_sys_tokens)
        concat_ids = self.text_inst_sys_tokens[:]

        split_memory_ids = []
        index = 0
        for size in each_mem_len:
            split_memory_ids.append(memory_ids[index:index + size])
            index += size

        for i in range(mem_num):
            tem_mem_id = split_memory_ids[i] + self.all_memory_sum_tokens[i]
            concat_ids += tem_mem_id
            segment_ids = segment_ids + [i + 1] * len(split_memory_ids[i]) + [0] * self.reencode_num

        concat_ids = concat_ids + user_tokens + remaining_ids
        segment_ids = segment_ids + [0] * (len(user_tokens) + len(remaining_ids))
        mem_len = mem_len + self.reencode_num *  mem_num
        labels = [-100] * (sys_len + mem_len + user_len) + remaining_ids

        if not len(concat_ids) == len(labels):
            print("concat_ids", len(concat_ids))
            print("labels", len(labels))
            print("Mem", mem_num, mem_len)
            print("concat_ids", len(remaining_ids))
            print(sys_len, mem_len, user_len)
            print('textinst')

        return {
            "input_ids": concat_ids,
            "labels": labels,
            "segment_ids": segment_ids,
        }

    def process_text(
        self,
        example: Dict[str, str],
    ):
        text_tokens = self.tokenizer(example["text"])['input_ids'][:self.max_len]
        labels = text_tokens
        segment_ids = [0] * len(text_tokens)
        return {
            "input_ids": text_tokens,
            "labels": labels,
            "segment_ids": segment_ids,
        }

    def process_textmem(
        self,
        example: Dict[str, str],
    ):
        sys_tokens = [self.bos_token_id, self.mem_start]
        sys_len = len(sys_tokens)

        user_tokens = [self.mem_end]
        user_len = len(user_tokens)

        text = example["text"]
        input_ids = self.tokenizer(text, add_special_tokens= False)['input_ids']

        input_ids = input_ids[:self.max_len - user_len - sys_len]

        mem_len = random.randint(self.min_memory_length, self.max_memory_length)
        mem_num = random.randint(self.min_memory_num, self.max_memory_num)

        breaks = sorted(random.sample(range(1, mem_len), mem_num - 1))
        breaks = [0] + breaks + [mem_len]
        each_mem_len = [breaks[i+1] - breaks[i] for i in range(mem_num)]

        # allocate space for special tokens
        input_len = len(input_ids)
        input_ids = input_ids[:input_len - self.reencode_num * mem_num]

        memory_ids = input_ids[:mem_len]
        remaining_ids = input_ids[mem_len:]
        concat_ids = sys_tokens

        split_memory_ids = []
        index = 0
        for mem_id, size in enumerate(each_mem_len):
            split_memory_ids.append(memory_ids[index:index + size])
            index += size

        segment_ids = [0] * sys_len
        for mem_id in range(mem_num):
            tem_mem_id = split_memory_ids[mem_id]
            for sub_idx in range(self.reencode_num):
                tem_mem_id = tem_mem_id + [self.special_token_start + self.reencode_num * mem_id + sub_idx]
            segment_ids = segment_ids + [mem_id + 1] * each_mem_len[mem_id] + [0] * self.reencode_num
            concat_ids += tem_mem_id

        concat_ids = concat_ids + user_tokens + remaining_ids
        mem_len = mem_len + self.reencode_num *  mem_num
        labels = [-100] * (sys_len + mem_len + user_len) + remaining_ids
        segment_ids = segment_ids + [0] * len(user_tokens) + [0] * len(remaining_ids)

        return {
            "input_ids": concat_ids,
            "labels": labels,
            "segment_ids": segment_ids,
        }

    def process_qamem(
        self,
        example: Dict[str, str],
    ):
        qa_system_input_ids = random.choice(self.qa_system_input_id_list)
        input_ids = qa_system_input_ids[:] + [self.mem_start]
        segment_ids = [0] * len(input_ids)

        formated_input_text_list = [
            f"Document [{j+1}](Title: {example['documents'][j]['title']}) "
            f"{example['documents'][j]['text']}\n" for j in range(10)
        ]
        if self.use_hf_tokenizer:
            formated_input_ids = self.tokenizer(
                formated_input_text_list,
                add_special_tokens=False,
                padding=False,
                truncation=False,
                return_attention_mask=False,
                return_token_type_ids=False,
                return_offsets_mapping=False,
                return_special_tokens_mask=False,
                return_length=False,
            )["input_ids"]
        else:
            formated_input_ids = [
                self.tokenizer(x, add_special_tokens=False)["input_ids"]
                for x in formated_input_text_list
            ]

        for idx in range(self.qa_document_num):
            input_ids = input_ids + formated_input_ids[idx] + self.all_memory_sum_tokens[idx]
            segment_ids = segment_ids + [idx + 1] * len(formated_input_ids[idx]) + [0] * self.reencode_num

        user_input_ids = [self.mem_end] + [self.eot_token_id] + self.user_start_token_ids + self.tokenizer(example['question'], add_special_tokens=False)["input_ids"] + [self.eot_token_id] + self.assistant_start_token_ids
        input_ids = input_ids + user_input_ids
        segment_ids = segment_ids + [0] * len(user_input_ids)

        ans_id = self.tokenizer(example["generated"] + "<|eot_id|>", add_special_tokens=False)["input_ids"]
        input_ids += ans_id
        segment_ids = segment_ids + [0] * len(ans_id)

        ans_len = len(ans_id)
        input_len = len(input_ids)

        labels = [-100] * (input_len - ans_len) + ans_id

        return {
            "input_ids": input_ids,
            "labels": labels,
            "segment_ids": segment_ids,
        }

    def process_qa(
        self,
        example: Dict[str, str],
    ):
        qa_system_input_ids = random.choice(self.qa_system_input_id_list)
        system_input_ids = qa_system_input_ids
        input_ids = system_input_ids

        for j in range(self.qa_document_num):
            title = example['documents'][j]['title']
            text = example['documents'][j]['text']
            tem_id = self.tokenizer(f"Document [{j+1}](Title: {title}) {text}\n", add_special_tokens=False)["input_ids"]

            input_ids += tem_id

        user_input_ids = [self.eot_token_id] + self.user_start_token_ids + self.tokenizer(example['question'], add_special_tokens=False)["input_ids"] + [self.eot_token_id] + self.assistant_start_token_ids
        input_ids += user_input_ids

        ans_id = self.tokenizer(example['generated'] + "<|eot_id|>", add_special_tokens=False)["input_ids"]
        input_ids += ans_id

        ans_len = len(ans_id)
        input_len = len(input_ids)

        labels = [-100] * (input_len - ans_len) + ans_id

        segment_ids = [0] * len(input_ids)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "segment_ids": segment_ids
        }

    def process_tulu(
        self,
        example: Dict[str, str],
    ):
        conversation = example["messages"]
        input_texts = [conversation[i]["content"] for i in range(len(conversation))]
        if self.use_hf_tokenizer:
            all_conversation_texts_ids = self.tokenizer(
                input_texts,
                add_special_tokens=False,
                padding=False,
                truncation=False,
                return_attention_mask=False,
                return_token_type_ids=False,
                return_offsets_mapping=False,
                return_special_tokens_mask=False,
                return_length=False,
            )["input_ids"]
        else:
            all_conversation_texts_ids = [
                self.tokenizer(x, add_special_tokens=False)["input_ids"] for x in input_texts
            ]

        tulu_system_input_ids = random.choice(self.sft_system_input_id_list)
        input_ids = tulu_system_input_ids
        labels = [-100] * len(input_ids)

        for i in range(len(conversation)):

            if conversation[i]["role"] == "user":
                user_msg_input_ids = (
                    self.user_start_token_ids + all_conversation_texts_ids[i]
                    + [self.eot_token_id]
                )
                if len(labels) + len(user_msg_input_ids) >= self.max_len:
                    break

                labels.extend([-100] * len(user_msg_input_ids))
                input_ids += user_msg_input_ids

            elif conversation[i]["role"] == "assistant":
                assist_msg_input_ids = (
                    self.assistant_start_token_ids + all_conversation_texts_ids[i]
                )
                if len(labels) + len(assist_msg_input_ids) > self.max_len - 1:
                    assist_msg_input_ids = input_ids[:self.max_len - 1 - len(labels)]

                assist_msg_input_ids += [self.eot_token_id]
                labels.extend(assist_msg_input_ids)
                input_ids += assist_msg_input_ids

        segment_ids = [0] * len(input_ids)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "segment_ids": segment_ids
        }

    def process_xsum(
        self,
        example: Dict[str, str],
    ):
        text_ids = self.tokenizer(example['document'], add_special_tokens=False)["input_ids"]
        if len(text_ids) > 1900:
            chunk_size = 220
        else:
            chunk_size = 100
        chunks = [text_ids[i:i+chunk_size] for i in range(0, len(text_ids), chunk_size)]

        xsum_system_input_ids = random.choice(self.summary_system_input_id_list)
        input_ids = xsum_system_input_ids + [self.mem_start]
        segment_ids = [0] * len(input_ids)

        for j in range(len(chunks)):
            input_ids = input_ids + chunks[j] + self.all_memory_sum_tokens[j]
            segment_ids = segment_ids + [j+1] * len(chunks[j]) + [0] * self.reencode_num

        ans_id = self.tokenizer(
            example['summary'] + "<|eot_id|>",
            add_special_tokens=False,
        )["input_ids"]
        assistant_input_ids = (
            [self.mem_end, self.eot_token_id] + self.assistant_start_token_ids
            + ans_id
        )

        input_ids = input_ids + assistant_input_ids
        labels = [-100] * (len(input_ids) - len(ans_id)) + ans_id
        segment_ids = segment_ids + [0] * len(assistant_input_ids)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "segment_ids": segment_ids
        }

@dataclass
class BlockAttnCollator():
    pad_token_idx: int = 124004

    def __call__(self, features: List[Dict[str, Any]]):
        input_ids = []
        labels = []
        segment_ids = []
        length_list = [len(x['input_ids']) for x in features]
        max_length = max(length_list)
        for idx in range(len(features)):
            seq_length = len(features[idx]['input_ids'])
            residual = max_length - seq_length
            input_ids.append(features[idx]['input_ids'] + [self.pad_token_idx] * residual)
            labels.append(features[idx]['labels'] + [-100] * residual)
            segment_ids.append(features[idx]["segment_ids"] + [-1] * residual)

        input_ids = torch.LongTensor(input_ids)
        labels = torch.LongTensor(labels)
        segment_ids = torch.LongTensor(segment_ids)

        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "segment_ids": segment_ids,
        }
        return batch

def _make_bool_segment_mask(
    *,
    source_segments: torch.Tensor,
    target_segments: torch.Tensor,
) -> torch.Tensor:
    target_segments = target_segments.unsqueeze(-1)
    source_segments = source_segments.unsqueeze(-2)

    # Returning the boolean mask based on equality
    return torch.eq(source_segments, target_segments)[:, ...]


def make_segment_mask(
    *,
    source_segments: torch.Tensor,
    target_segments: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
    add_causal_lm_mask: bool = True,
) -> torch.Tensor:
    """Generates attention logit biases given the segment ids.

    ... such that positions belonging to different segments cannot attend to each other.
    This function is based on the implementation of AXLearn from Apple. The original
        implementation of the `make_segment_mask` can be found in:
        https://github.com/apple/axlearn/blob/main/axlearn/common/attention_bias.py#L767

    Args:
        source_segments: An integer tensor of shape [batch, ..., source_length].
        target_segments: An integer tensor of shape [batch, ..., target_length].

    Returns:
        A float Tensor of shape [batch, 1, ..., target_length, source_length] where the
        value at [..., i, j] = 0 if target_segments[..., i] == source_segments[..., j], or -inf
        otherwise.
    """
    # min_dtype = torch.finfo(dtype).min
    min_dtype = -float("inf")
    sequence_len = source_segments.size(-1)
    batch_size = source_segments.size(0)
    if add_causal_lm_mask:
        segment_logit_bias = torch.triu(
            torch.full(
                (batch_size, sequence_len, sequence_len),
                # NEG_INF,
                min_dtype,
                dtype=dtype,
                device=source_segments.device
            ),
        diagonal=1)
    else:
        segment_logit_bias = torch.zeros(
            size=(batch_size, sequence_len, sequence_len),
            dtype=dtype,
            device=source_segments.device,
        )

    # within the same segment
    bool_mask = _make_bool_segment_mask(
        source_segments=source_segments, target_segments=target_segments
    )

    # Create masks for tokens belonging to segment 0
    # Shape [batch, ..., 1, source_length]
    target_is_zero = (target_segments == 0).unsqueeze(-1)

    # Tokens in segment 0 can be attended by any token
    zero_mask = target_is_zero

    # masks that indicates the token is a pad token
    # Shape [batch, ..., 1, source_length]
    source_invalid_mask = (source_segments == -1).unsqueeze(-2)
    target_invalid_mask = (target_segments == -1).unsqueeze(-1)
    # Combine invalid masks: pad tokens
    invalid_mask = source_invalid_mask | target_invalid_mask

    # all_masks = (~bool_mask) & (~zero_mask)
    all_masks = invalid_mask | ((~bool_mask) & (~zero_mask))
    segment_logit_bias = segment_logit_bias.masked_fill_(all_masks, min_dtype)

    # if dtype is torch.bfloat16:
    #     segment_logit_bias = segment_logit_bias.bfloat16()
    return segment_logit_bias

def custom_collate_bias(batch):
    input_ids = []
    labels = []
    biased_index = []
    mem_num = []
    input_length = []
    for item in batch:
        if item['biased_index'] is not None:
            mem_num.append(len(item['biased_index']))
        else:
            mem_num.append(0)
        input_length.append(len(item['input_ids']))

    max_mem_num = max(mem_num)
    max_length = max(input_length)

    for item in batch:
        seq_length = len(item['input_ids'])
        _mem_num = len(item['biased_index']) if item['biased_index'] is not None else 0
        input_ids.append(item['input_ids'] + [0] * (max_length - seq_length))
        labels.append(item['labels'] + [-100] * (max_length - seq_length))

        if item['biased_index'] is None:
            curr_biased_index =  [[0,0]] * (max_mem_num)
            biased_index.append(curr_biased_index)
        else:
            biased_index.append(item['biased_index'] + [[0,0]] * (max_mem_num - _mem_num))

    return {
        'input_ids': torch.LongTensor(input_ids),
        'labels': torch.LongTensor(labels),
        # 'labels': torch.LongTensor(input_ids),
        'biased_index': torch.LongTensor(biased_index),
        "input_length": torch.LongTensor(input_length),
        'mem_num': torch.LongTensor(mem_num),
    }


