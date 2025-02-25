"""
Loading dataset in a streaming way and allow resume from checkpoints
Reference: torchtitan/datasets/hf_dataset.py
"""
import pickle
import random
from typing import Any, Dict, List, Tuple

import datasets
from datasets.distributed import split_dataset_by_node
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset
from torchdata.stateful_dataloader import StatefulDataLoader

from src.data.titan_preprocessor import (
    SumAttentionPreprocessor,
)
from src.data.titan_tokenizer import Tokenizer
from src.torchtitan.logging import logger


def load_data_and_process_fn(
    data_component_name: str,
    preprocessor: SumAttentionPreprocessor,
    training: bool = True,
) -> Tuple[datasets.IterableDataset, datasets.Dataset]:
    """
    load the downloaded data from disk and then pair it with the preprocessor
    """
    if data_component_name in ["text", "text_mem", "text_inst"]:
        data_path = f"dataset_cache/processed/fineweb/{data_component_name}"
        if data_component_name == "text":
            preprocessor_fn = preprocessor.process_text
        elif data_component_name == "text_mem":
            preprocessor_fn = preprocessor.process_textmem
        elif data_component_name == "text_inst":
            preprocessor_fn = preprocessor.process_textinst
        else:
            raise NotImplementedError()
        remove_columns = [
            "text", "id", "dump", "url", "date",
            "file_path", "language", "language_score", "token_count",
        ]
        if data_component_name in ["text_mem", "text_inst"]:
            remove_columns.append("num_tokens")
        num_shards = 512
    elif data_component_name in ["sft", "sft_mem"]:
        data_path = f"dataset_cache/processed/daringanteater/{data_component_name}"
        if data_component_name == "sft":
            preprocessor_fn = preprocessor.process_sft
        elif data_component_name == "sft_mem":
            preprocessor_fn = preprocessor.process_sftmem
        else:
            raise NotImplementedError()
        remove_columns=["system", "mask", "dataset", "conversations"]
        num_shards = 32
    elif data_component_name in ["tulu"]:
        data_path = "dataset_cache/processed/tulu/sft"
        if data_component_name == "tulu":
            preprocessor_fn = preprocessor.process_tulu
        else:
            raise NotImplementedError()
        remove_columns=["id", "messages", "source"]
        num_shards = 32
    elif data_component_name in ["qa", "qa_mem"]:
        data_path = f"dataset_cache/processed/block_qa/{data_component_name}"
        if data_component_name == "qa":
            preprocessor_fn = preprocessor.process_qa
        elif data_component_name == "qa_mem":
            preprocessor_fn = preprocessor.process_qamem
        else:
            raise NotImplementedError()
        remove_columns=['prompt', 'question', 'answers', 'generated', 'inputs', 'documents']
        num_shards = 32
    elif data_component_name in ["xsum"]:
        data_path = f"dataset_cache/processed/xsum/{data_component_name}"
        preprocessor_fn = preprocessor.process_xsum
        remove_columns=['document', 'summary', 'id']
        num_shards = 32
    else:
        raise ValueError(f"Unrecognized dataset name {data_component_name}.")

    data_component: datasets.DatasetDict = datasets.load_from_disk(data_path)
    if training:
        ds = data_component["train"].to_iterable_dataset(num_shards=num_shards)
    else:
        ds = data_component["test"]
    return ds, preprocessor_fn, remove_columns

class HuggingFaceDataset(IterableDataset, Stateful):
    def __init__(
        self,
        dataset_name: str,
        tokenizer: Tokenizer,
        preprocessor: SumAttentionPreprocessor,
        seq_len: int = 4096,
        world_size: int = 1,
        rank: int = 0,
        infinite: bool = False,
        packing_mode: str = "padding",
    ) -> None:
        self.packing_mode = packing_mode
        assert self.packing_mode in ["padding", "packing"]
        ds, preprocess_fn, columns_to_remove = load_data_and_process_fn(
            data_component_name=dataset_name,
            preprocessor=preprocessor,
            training=True,
        )

        self.dataset_name = dataset_name
        self._data = split_dataset_by_node(ds, rank, world_size)
        self._tokenizer = tokenizer
        self.seq_len = seq_len
        self.infinite = infinite
        self.preprocess_fn = preprocess_fn
        self.columns_to_remove = columns_to_remove

        # Variables for checkpointing
        self._sample_idx = 0
        self._all_tokens: List[int] = []
        self._all_labels: List[int] = []

    def _get_data_iter(self):
        if self._sample_idx == 0:
            return iter(self._data)

        if isinstance(self._data, datasets.Dataset) and self._sample_idx == len(self._data):
            return iter([])

        return iter(self._data.skip(self._sample_idx))

    def __iter__(self):
        # Original Titan Trainer write it as `1 + self.seq_len` because it will shift the labels
        # by 1. But for our case (huggingface models), the shift happens when computing the loss
        # So, no need to +1 here
        max_buffer_token_len = self.seq_len

        while True:
            for sample in self._get_data_iter():
                # Use the dataset-specific text processor
                processed_example = self.preprocess_fn(sample)
                if self.packing_mode == "padding":
                    yield processed_example
                    self._sample_idx += 1
                else:
                    input_ids = processed_example["input_ids"]
                    labels = processed_example["labels"]
                    self._all_tokens.extend(input_ids)
                    self._all_labels.extend(labels)
                    self._sample_idx += 1

                    while len(self._all_tokens) >= max_buffer_token_len:
                        packed_tokens = self._all_tokens[:max_buffer_token_len]
                        packed_labels = self._all_labels[:max_buffer_token_len]
                        # update tokens to the remaining tokens
                        self._all_tokens = self._all_tokens[max_buffer_token_len:]
                        self._all_labels = self._all_labels[max_buffer_token_len:]
                        packed_example = {
                            "input_ids": packed_tokens,
                            "labels": packed_labels,
                            "biased_index": None,
                        }
                        yield packed_example


            if not self.infinite:
                logger.warning(f"Dataset {self.dataset_name} has run out of data")
                break
            else:
                # Reset offset for the next iteration
                self._sample_idx = 0
                logger.warning(f"Dataset {self.dataset_name} is being re-looped")

    def load_state_dict(self, state_dict):
        self._sample_idx = state_dict["sample_idx"]
        if self.packing_mode == "packing":
            self._all_tokens = state_dict["token_buffer"]
            self._all_labels = state_dict["label_buffer"]

    def state_dict(self):
        return {
            "token_buffer": self._all_tokens,
            "label_buffer": self._all_labels,
            "sample_idx": self._sample_idx,
        }
        # return {"sample_idx": self._sample_idx}


class WeightedAggregatorDataset(IterableDataset, Stateful):
    """
    This dataset randomly chooses one child dataset to sample from at each iteration,
    according to the provided weights. Each child must be an IterableDataset that also
    implements Stateful, so we can maintain and restore states properly.
    """
    def __init__(
        self,
        datasets: List[IterableDataset],
        weights: List[float],
        seed: int = 42,
        infinite: bool = True,
    ):
        """
        Args:
            datasets: List of child IterableDatasets (each must be Stateful).
            weights: Sampling probability weights (normalized internally).
            seed: Seed for random choice.
            infinite: If True, cycle forever. Otherwise, you'll eventually exhaust all children.
        """
        super().__init__()
        assert len(datasets) == len(weights), "datasets and weights must be the same length"
        self.datasets = datasets
        self.weights = weights
        self.infinite = infinite

        # Normalize weights
        total = sum(weights)
        self.probs = [w / total for w in weights]

        # We'll hold iterators for each dataset
        self.iters = [None] * len(datasets)

        # For checkpointing
        self._rng_state = None
        self._random_gen = random.Random(seed)

    def state_dict(self):
        """
        Return aggregator's internal state + each child's state.
        """
        state = {
            "rng_state": self._rng_state,
            "children": {},
        }
        for i, ds in enumerate(self.datasets):
            # Each child is also Stateful
            state["children"][i] = ds.state_dict()
        return state

    def load_state_dict(self, state_dict):
        """
        Load aggregator state + each child's state.
        """
        if not state_dict:
            return

        # Restore aggregator RNG state
        self._rng_state = state_dict["rng_state"]
        if self._rng_state is not None:
            self._random_gen.setstate(self._rng_state)

        # Restore each child's state
        for i, ds in enumerate(self.datasets):
            ds.load_state_dict(state_dict["children"][i])

    def __iter__(self):
        """
        Main iterator that picks which child to sample from using the aggregator’s probabilities.
        """
        # Create child iterators if needed
        self.iters = [iter(ds) for ds in self.datasets]

        while True:
            # Use aggregator’s random choice
            choice = self._random_gen.choices(range(len(self.datasets)), weights=self.probs, k=1)[0]

            try:
                sample = next(self.iters[choice])
                yield sample
            except StopIteration:
                # If one dataset is exhausted and infinite=False, you must decide how to handle it.
                # If infinite=True, you might re-create its iterator, or skip it, etc.
                # For now, we’ll just break (or re-loop) for demonstration.
                if not self.infinite:
                    break
                else:
                    self.iters[choice] = iter(self.datasets[choice])
                    continue

    def __next__(self):
        return next(iter(self))

    def __del__(self):
        # If you need to clean up any references
        pass

    def _save_rng_state(self):
        """Helper to capture aggregator's RNG state if needed."""
        self._rng_state = self._random_gen.getstate()


class DPAwareDataLoader(StatefulDataLoader, Stateful):
    """
    A wrapper around the StatefulDataLoader that ensures that the state is stored only once per DP rank.
    """

    def __init__(
        self,
        dp_rank: int,
        hf_ds: IterableDataset,
        batch_size: int,
        collate_fn,
    ):
        super().__init__(hf_ds, batch_size, collate_fn=collate_fn)
        self._dp_rank = dp_rank
        self._rank_id = f"dp_rank_{dp_rank}"

    def state_dict(self) -> Dict[str, Any]:
        # Store state only for dp rank to avoid replicating the same state across other dimensions
        return {self._rank_id: pickle.dumps(super().state_dict())}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        # State being empty is valid
        if not state_dict:
            return

        if self._rank_id not in state_dict:
            logger.warning(
                f"DataLoader state is empty for dp rank {self._dp_rank}, expected key {self._rank_id}"
            )
            return
        super().load_state_dict(pickle.loads(state_dict[self._rank_id]))






