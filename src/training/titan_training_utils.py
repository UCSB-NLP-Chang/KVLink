from dataclasses import replace
from typing import List

from src.training.titan_trainer_config_utils import (
    ActivationCheckpoint,
    ActivationCheckpointMode,
    CheckpointConfig,
    DataComponent,
    TrainingRecipe,
)

DATA_ORIGINAL: List[DataComponent] = [
    DataComponent(dataset_name="text", weight=20.0),
    DataComponent(dataset_name="tulu", weight=30.0),
    DataComponent(dataset_name="sft_mem", weight=25.0),
    DataComponent(dataset_name="qa", weight=10.0),
    DataComponent(dataset_name="qa_mem", weight=10.0),
    DataComponent(dataset_name="xsum", weight=5.0),
]

DATA_NOSUM: List[DataComponent] = [
    DataComponent(dataset_name="text", weight=21.1),
    DataComponent(dataset_name="tulu", weight=31.6),
    DataComponent(dataset_name="sft_mem", weight=26.3),
    DataComponent(dataset_name="qa", weight=10.5),
    DataComponent(dataset_name="qa_mem", weight=10.5),
]

DATA_NOSFTMEM: List[DataComponent] = [
    DataComponent(dataset_name="text", weight=26.7),
    DataComponent(dataset_name="tulu", weight=40.0),
    DataComponent(dataset_name="qa", weight=13.3),
    DataComponent(dataset_name="qa_mem", weight=13.3),
    DataComponent(dataset_name="xsum", weight=6.7),
]

DATA_QAONLY: List[DataComponent] = [
    DataComponent(dataset_name="qa_mem", weight=100.0)
]

DATASET_MAPPING = {
    "original": DATA_ORIGINAL,
    "nosum": DATA_NOSUM,
    "nosftmem": DATA_NOSFTMEM,
    "qaonly": DATA_QAONLY
}

COMMON_CHECKPOINT_CONFIG = CheckpointConfig(
    enable_checkpoint=True,
    folder="checkpoints",
    interval_type="step",
    interval=500,
    model_weights_only=False,
    export_dtype="bfloat16",
    create_seed_checkpoint=False,
    async_mode="disabled",
    keep_latest_k=2,
    load_step=-1,
)

DEFUALT_TRAINING_RECIPE = TrainingRecipe(
    batch_size=32,
    lr=5e-6,
    max_steps=10_000,
    warmup_steps=1_000,
    fused=False,
    max_norm=1.0,
    eval_every_n_steps=1000,
)

bsz64_lr56_steps6k =replace(
    DEFUALT_TRAINING_RECIPE,
    batch_size=64,
    max_steps=6000,
    warmup_steps=600,
    eval_every_n_steps=500,
)

bsz64_lr56_steps600 =replace(
    DEFUALT_TRAINING_RECIPE,
    batch_size=64,
    max_steps=600,
    warmup_steps=60,
    eval_every_n_steps=100,
)

TRAINING_RECIPE_MAPS = {
    "bsz32_lr25_steps10k": DEFUALT_TRAINING_RECIPE,
    "bsz64_lr56_steps6k": bsz64_lr56_steps6k
}


FULL_ACTIVATION_CHECKPOINT_CONFIG = ActivationCheckpoint(
    mode=ActivationCheckpointMode.FULL,
    selective_ac_option="op",
    # selective_ac_option="2",
)

SELECTIVE_ACTIVATION_CHECKPOINT_CONFIG = ActivationCheckpoint(
    mode=ActivationCheckpointMode.SELECTIVE,
    selective_ac_option="op",
    # selective_ac_option="2",
)

PRETRAINED_MODEL_CKPT_PATH_MAPS = {
    "meta-llama/Llama-3.2-1B-Instruct": "model_cache/Llama-3.2-1B-Instruct/model.safetensors",
}
