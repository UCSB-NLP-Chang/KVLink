from dataclasses import dataclass
from enum import Enum


class ActivationCheckpointMode(Enum):
    NONE = "none"
    SELECTIVE = "selective"
    FULL = "full"

@dataclass
class ActivationCheckpoint:
    mode: ActivationCheckpointMode
    selective_ac_option: str


@dataclass
class CheckpointConfig:
    enable_checkpoint: bool
    folder: str
    interval_type: str
    interval: int
    model_weights_only: bool
    export_dtype: str
    create_seed_checkpoint: bool
    async_mode: str
    keep_latest_k: int
    load_step: int


@dataclass
class CommonConfig:
    gc_freq: int = 50
    seed: int = 42
    deterministic: bool = False


@dataclass
class TrainingRecipe:
    batch_size: int
    lr: float
    max_steps: int
    warmup_steps: int
    fused: bool
    max_norm: float
    eval_every_n_steps: int


@dataclass
class TitanTrainerConfig:
    model_name_or_path: str
    tokenizer_path: str
    dataset_version: str
    seq_len: int
    job_dump_folder: str
    activation_checkpoint: ActivationCheckpoint
    ckpt_config: CheckpointConfig
    training_recipe: TrainingRecipe
    # TODO: (KVLinkDeveloper) temporarily add it here. Need to refactor the code later
    enable_packing: bool = False
    # TODO: (KVLinkDeveloper) temporarily add it here. Need to refactor the code later
    reencode_num: int = 0
    max_memory_num: int = 40
    pass


@dataclass
class DataComponent:
    dataset_name: str
    weight: float


