import torch
from transformers import Trainer

from src.data.attention import construct_biased_attention_matrix


class CustomTrainerBiasAttn(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        attention_matrices = []
        max_length = max(inputs['input_length'])
        for idx in range(len(inputs['input_ids'])):
            mem_num = inputs['mem_num'][idx]
            if mem_num == 0:
                biased_ranges = None
            else:
                biased_ranges = inputs['biased_index'][idx][:mem_num]
            attention_matrices.append(
                construct_biased_attention_matrix(
                    inputs['input_length'][idx],
                    biased_ranges,
                    max_length,
                    inputs['input_ids'].device
                ).unsqueeze(0)
            )

        outputs = model(input_ids = inputs['input_ids'], attention_mask = torch.stack(attention_matrices), labels = inputs['labels'])
        # outputs = model(input_ids = inputs['input_ids'], labels = inputs['labels'])

        return (outputs.loss, outputs) if return_outputs else outputs.loss
