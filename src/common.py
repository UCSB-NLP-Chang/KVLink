from typing import Any, Dict, List, TypeVar, Union, overload

import torch
from transformers.tokenization_utils_base import BatchEncoding

T = TypeVar('T', bound=Union[torch.Tensor, BatchEncoding, dict, list, Any])

@overload
def move_to_target_device(object: torch.Tensor, device: torch.device) -> torch.Tensor: ...
@overload
def move_to_target_device(object: BatchEncoding, device: torch.device) -> BatchEncoding: ...
@overload
def move_to_target_device(object: Dict, device: torch.device) -> Dict: ...
@overload
def move_to_target_device(object: List, device: torch.device) -> List: ...
@overload
def move_to_target_device(object: Any, device: torch.device) -> Any: ...

def move_to_target_device(object: T, device: torch.device) -> T:
    if torch.is_tensor(object):
        return object.to(device)
    elif isinstance(object, dict):
        return {k: move_to_target_device(v, device) for k, v in object.items()}
    elif isinstance(object, BatchEncoding):
        return {k: move_to_target_device(v, device) for k, v in object.items()}
    elif isinstance(object, list):
        return [move_to_target_device(x, device) for x in object]
    else:
        return object
