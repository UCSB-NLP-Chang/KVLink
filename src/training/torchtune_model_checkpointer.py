"""
Load pre-trained Llama model weights following torchtune.
"""
import gc
from typing import Any, Dict

import torch
from torchtune.models import convert_weights
from torchtune.training.checkpointing._utils import (
    safe_torch_load,
)

MODEL_CONFIG_DICT = {
    "meta-llama/Llama-3.2-1B-Instruct": {
        "num_attention_heads": 32,
        "num_hidden_layers": 16,
        "num_key_value_heads": 8,
        "hidden_size": 2048,
        "head_dim": 64,
    }
}

def load_checkpoint(
    ckpt_path: str,
    model_name: str,
) -> Dict[str, Any]:
    """
    Load HF checkpoint from file.

    The keys and weights from across all checkpoint files are merged into a single state_dict.
    We preserve the "state_dict key" <-> "checkpoint file" mapping in weight_map so we can
    write the state dict correctly in ``save_checkpoint``.

    Before returning, the model state dict is converted to a torchtune-compatible format using
    the appropriate convert_weights function (depending on ``self._model_type``).

    Args:
        ckpt_path: the path that store the model checkpoint downloaded from huggingface `original`
        model_name: the HF model name, such as `meta-llama/Llama-3.2-1B-Instruct`
    Returns:
        state_dict (Dict[str, Any]): torchtune checkpoint state dict

    Raises:
        ValueError: If the values in the input state_dict are not Tensors
    """

    # merged state_dict contains keys and weights from all the checkpoint files
    merged_state_dict: Dict[str, torch.Tensor] = {}

    # converted_state_dict is the final state_dict passed to the recipe after the
    # keys are converted into the torchtune format. This optionally also contains
    # the recipe state and adapter weights
    converted_state_dict: Dict[str, Dict[str, torch.Tensor]] = {}

    # _checkpoint_paths are already sorted so simply enumerate to generate the right id
    state_dict = safe_torch_load(ckpt_path)
    merged_state_dict.update(state_dict)

    # delete the state_dict to free up memory; TODO check if this del is needed
    del state_dict
    gc.collect()

    cfg_dict = MODEL_CONFIG_DICT[model_name]
    converted_state_dict = convert_weights.hf_to_tune(
        merged_state_dict,
        num_heads=cfg_dict["num_attention_heads"],
        num_kv_heads=cfg_dict["num_key_value_heads"],
        dim=cfg_dict["hidden_size"],
        head_dim=cfg_dict.get("head_dim", None),
    )

    return converted_state_dict



