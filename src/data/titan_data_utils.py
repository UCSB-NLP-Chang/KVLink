from typing import List

from torch.utils.data import DataLoader, DistributedSampler

from src.data.titan_datasets import (
    DPAwareDataLoader,
    HuggingFaceDataset,
    WeightedAggregatorDataset,
    load_data_and_process_fn,
)
from src.data.titan_preprocessor import (
    SumAttentionPreprocessor,
)
from src.data.titan_tokenizer import LLaMA32Tokenizer
from src.training.titan_trainer_config_utils import (
    DataComponent,
)


def build_hf_data_loader(
    data_components: List[DataComponent],
    tokenizer: LLaMA32Tokenizer,
    preprocessor: SumAttentionPreprocessor,
    seed: int,
    batch_size: int,
    seq_len: int,
    world_size: int,
    rank: int,
    collate_fn,
    infinite: bool = True,
    enable_packing: bool = False,
):
    """Build a data loader for HuggingFace datasets."""
    all_datasets = []
    dataset_weights = []
    for data_component in data_components:
        dataset_name = data_component.dataset_name
        weight = data_component.weight
        if enable_packing:
            packing_mode = "packing" if dataset_name in ["sft", "tulu", "text"] else "padding"
        else:
            packing_mode = "padding"
        hf_ds = HuggingFaceDataset(
            dataset_name,
            tokenizer,
            seq_len=seq_len,
            preprocessor=preprocessor,
            world_size=world_size,
            rank=rank,
            infinite=infinite,
            packing_mode=packing_mode,
        )
        all_datasets.append(hf_ds)
        dataset_weights.append(weight)

    combined_ds = WeightedAggregatorDataset(
        all_datasets,
        dataset_weights,
        seed=seed,
        infinite=infinite,
    )
    return DPAwareDataLoader(rank, combined_ds, batch_size=batch_size, collate_fn=collate_fn)


def build_hf_eval_data_loader(
    data_components: List[DataComponent],
    tokenizer: LLaMA32Tokenizer,
    preprocessor: SumAttentionPreprocessor,
    batch_size: int,
    seq_len: int,
    world_size: int,
    rank: int,
    collate_fn,
):
    """Build a data loader for HuggingFace datasets."""
    dataloader_dict = {}
    for data_component in data_components:
        dataset_name = data_component.dataset_name
        ds, preprocess_fn, columns_to_remove = load_data_and_process_fn(
            data_component_name=dataset_name,
            preprocessor=preprocessor,
            training=False,
        )
        ds = ds.map(preprocess_fn, num_proc=32,remove_columns=columns_to_remove,batched=False)

        sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=False)
        valid_loader = DataLoader(
            ds,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=collate_fn,
        )
        dataloader_dict[dataset_name] = valid_loader

    return dataloader_dict


