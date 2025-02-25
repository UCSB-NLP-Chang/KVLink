# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
1. Download the tokenizer
python src/data/titan_download_tokenizer.py \
    --repo_id meta-llama/Llama-3.2-1B-Instruct \
    --tokenizer_path "original" \
    --local_dir data/titan_tokenizer/ \
    --hf_token=YOUR_HF_TOKEN

2. Download the model
tune download meta-llama/Llama-3.2-1B-Instruct \
    --output-dir model_cache/Llama-3.2-1B-Instruct \
    --ignore-patterns "original/consolidated.00.pth" \
    --hf-token YOUR_HF_TOKEN \


3. Running

```sh
LOG_RANK=${LOG_RANK:-0}
NGPU=${NGPU:-"8"}

PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
torchrun --nproc_per_node=8 --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
    --local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
    titan_trainer.py --config_name block_datav1_step10k_bsz64_single_node_selective_ckpt
```

3. View the tensorboard logs in remote server:
Run the command in the local machine:
ssh -L 6006:localhost:6006 your_remote@remote_address

Run the command in the remote server:
tensorboard --logdir=/path/to/KVMemory/run_logs/block_datav1_step10k/tensorboard/20250105-1953 --port=6006

Now open http://localhost:6006/ in the local machine.

4. Visualize multiple tensorboard logs:
tensorboard --logdir_spec=full_ckpt:20250106-1910,selective_ckpt:20250106-1919 --port=6007
"""

import argparse
import os
import time
from datetime import timedelta
from typing import Dict

import torch
import torchtune.training as training
import tqdm
from torch.distributed.elastic.multiprocessing.errors import record
from torch.utils.data import DataLoader
from torchtune.models.llama3_2 import llama3_2_1b
from transformers import AutoTokenizer

from src.data.titan_data_utils import (
    SumAttentionPreprocessor,
    build_hf_data_loader,
    build_hf_eval_data_loader,
)
from src.data.titan_preprocessor import BlockAttnCollator, make_segment_mask
from src.data.titan_tokenizer import LLaMA32Tokenizer
from src.torchtitan import utils
from src.torchtitan.logging import init_logger, logger
from src.torchtitan.optimizer import build_lr_schedulers, build_optimizers
from src.torchtitan.profiling import maybe_enable_memory_snapshot, maybe_enable_profiling
from src.torchtitan.utils import device_module, device_type
from src.training.checkpointing import (
    CheckpointManager,
    TrainState,
)
from src.training.metrics import build_device_memory_monitor, build_metric_logger
from src.training.parallelism import (
    ParallelDims,
    parallelize_llama,
)
from src.training.titan_trainer_config_utils import (
    CommonConfig,
    TitanTrainerConfig,
)
from src.training.titan_training_utils import (
    COMMON_CHECKPOINT_CONFIG,
    DATASET_MAPPING,
    FULL_ACTIVATION_CHECKPOINT_CONFIG,
    PRETRAINED_MODEL_CKPT_PATH_MAPS,
    SELECTIVE_ACTIVATION_CHECKPOINT_CONFIG,
    bsz64_lr56_steps6k,
    bsz64_lr56_steps600
)
from src.training.torchtune_model_checkpointer import load_checkpoint

CONFIG_DICT = {
    "data_original_step6k_bsz64_link_5_selective_ckpt": TitanTrainerConfig(
        model_name_or_path="meta-llama/Llama-3.2-1B-Instruct",
        tokenizer_path="data/titan_tokenizer/original/tokenizer.model",
        dataset_version="original",
        seq_len=4096,
        reencode_num=5,
        job_dump_folder="run_logs/data_original_step6k_bsz64_link_5_selective_ckpt",
        ckpt_config=COMMON_CHECKPOINT_CONFIG,
        training_recipe=bsz64_lr56_steps6k,
        activation_checkpoint=SELECTIVE_ACTIVATION_CHECKPOINT_CONFIG,
    ),

    "data_original_step6k_bsz64_link_5_full_ckpt": TitanTrainerConfig(
        model_name_or_path="meta-llama/Llama-3.2-1B-Instruct",
        tokenizer_path="data/titan_tokenizer/original/tokenizer.model",
        dataset_version="original",
        seq_len=4096,
        reencode_num=5,
        job_dump_folder="run_logs/data_original_step6k_bsz64_link_5_full_ckpt",
        ckpt_config=COMMON_CHECKPOINT_CONFIG,
        training_recipe=bsz64_lr56_steps6k,
        activation_checkpoint=FULL_ACTIVATION_CHECKPOINT_CONFIG,
    ),

    "data_nosum_step6k_bsz64_link_5_full_ckpt": TitanTrainerConfig(
        model_name_or_path="meta-llama/Llama-3.2-1B-Instruct",
        tokenizer_path="data/titan_tokenizer/original/tokenizer.model",
        dataset_version="nosum",
        seq_len=4096,
        reencode_num=5,
        job_dump_folder="run_logs/data_nosum_step6k_bsz64_link_5_full_ckpt",
        ckpt_config=COMMON_CHECKPOINT_CONFIG,
        training_recipe=bsz64_lr56_steps6k,
        activation_checkpoint=FULL_ACTIVATION_CHECKPOINT_CONFIG,
    ),

    "data_nosftmem_step6k_bsz64_link_5_full_ckpt": TitanTrainerConfig(
        model_name_or_path="meta-llama/Llama-3.2-1B-Instruct",
        tokenizer_path="data/titan_tokenizer/original/tokenizer.model",
        dataset_version="nosftmem",
        seq_len=4096,
        reencode_num=5,
        job_dump_folder="run_logs/data_original_step6k_bsz64_link_5_full_ckpt",
        ckpt_config=COMMON_CHECKPOINT_CONFIG,
        training_recipe=bsz64_lr56_steps6k,
        activation_checkpoint=FULL_ACTIVATION_CHECKPOINT_CONFIG,
    ),

    "data_qaonly_step6k_bsz64_link_5_full_ckpt": TitanTrainerConfig(
        model_name_or_path="meta-llama/Llama-3.2-1B-Instruct",
        tokenizer_path="data/titan_tokenizer/original/tokenizer.model",
        dataset_version="qaonly",
        seq_len=4096,
        reencode_num=5,
        job_dump_folder="run_logs/data_original_step6k_bsz64_link_5_full_ckpt",
        ckpt_config=COMMON_CHECKPOINT_CONFIG,
        training_recipe=bsz64_lr56_steps600,
        activation_checkpoint=FULL_ACTIVATION_CHECKPOINT_CONFIG,
    )
}

# Enable debug tracing on failure: https://pytorch.org/docs/stable/elastic/errors.html
@record
def main(config_name: str, use_wandb_for_log: bool = False):
    init_logger()
    task_config = CONFIG_DICT[config_name]
    common_cfg = CommonConfig()

    job_dump_folder = task_config.job_dump_folder
    log_freq = 10
    train_timeout_seconds = 100
    eval_interval = task_config.training_recipe.eval_every_n_steps
    scheduler_type = "cosine"
    enable_data_packing = task_config.enable_packing
    # Empicially this is faster than the tokenizer from torchtitan
    use_hf_tokenizer = True


    logger.info(f"Starting job: {config_name}")

    logger.info(f"Running with args:\n{task_config}")

    # used for colorful printing
    color = utils.Color

    # take control of garbage collection to avoid stragglers
    gc_handler = utils.GarbageCollection(gc_freq=common_cfg.gc_freq)

    # init distributed
    world_size = int(os.environ["WORLD_SIZE"])
    parallel_dims = ParallelDims(
        # dp_shard=1,
        # dp_replicate=8,
        dp_shard=-1,
        dp_replicate=1,
        cp=1,
        tp=1,
        pp=1,
        world_size=world_size,
        enable_loss_parallel=False,
    )
    device = torch.device(f"{device_type}:{int(os.environ['LOCAL_RANK'])}")
    device_module.set_device(device)
    utils.init_distributed(
        job_dump_folder=job_dump_folder
    )
    # initialize device memory monitor and get peak flops for MFU calculation
    device_memory_monitor = build_device_memory_monitor()
    gpu_peak_flops = utils.get_peak_flops(device_memory_monitor.device_name)
    logger.info(f"Peak FLOPS used for computing MFU: {gpu_peak_flops:.3e}")

    # build meshes
    world_mesh = parallel_dims.build_mesh(device_type=device_type)
    if parallel_dims.dp_enabled:
        dp_mesh = world_mesh["dp"]
        dp_degree, dp_rank = dp_mesh.size(), dp_mesh.get_local_rank()
    else:
        dp_degree, dp_rank = 1, 0
    local_batch_size = task_config.training_recipe.batch_size // dp_degree


    if parallel_dims.pp_enabled:
        pp_mesh = world_mesh["pp"]

    # Set random seed, and maybe enable deterministic mode (mainly for debugging, expect perf loss)
    utils.set_determinism(
        world_mesh, device, common_cfg.seed, common_cfg.deterministic
    )
    model_name = task_config.model_name_or_path
    tokenizer_path = task_config.tokenizer_path

    # build tokenizer
    if use_hf_tokenizer:
        tokenizer = LLaMA32Tokenizer(tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    # build dataloader
    data_components = DATASET_MAPPING[task_config.dataset_version]
    data_collator = BlockAttnCollator(pad_token_idx=tokenizer.pad_id)
    preprocessor = SumAttentionPreprocessor(
        tokenizer=tokenizer,
        max_len=task_config.seq_len,
        special_token_start=128011,
        mem_start=128254,
        mem_end=128255,
        reencode_num=task_config.reencode_num,
        max_memory_num=task_config.max_memory_num,
    )
    data_loader = build_hf_data_loader(
        data_components,
        tokenizer,
        preprocessor=preprocessor,
        seed=common_cfg.seed,
        batch_size=local_batch_size,
        seq_len=task_config.seq_len,
        world_size=dp_degree,
        rank=dp_rank,
        infinite=True,
        collate_fn=data_collator,
        enable_packing=enable_data_packing,
    )
    eval_data_loader_dict: Dict[str, DataLoader] = build_hf_eval_data_loader(
        data_components,
        tokenizer,
        preprocessor=preprocessor,
        batch_size=local_batch_size,
        seq_len=task_config.seq_len,
        world_size=dp_degree,
        rank=dp_rank,
        collate_fn=data_collator,
    )
    if enable_data_packing:
        logger.info(
            "Note: Packing mode enabled."
            "Dataset that supported packing will be packed instead of padded."
        )

    logger.info(f"Building {model_name}...")
    with torch.device("meta"):
        model = llama3_2_1b()
    # log model size
    model_param_count = utils.get_num_params(model)
    logger.info(
        f"{color.blue}Model {model_name}"
        f"{color.red}size: {model_param_count:,} total parameters{color.reset}"
    )

    # loss function to be shared by Pipeline Parallel and SPMD training
    def loss_fn(pred, labels):
        return torch.nn.functional.cross_entropy(
            pred.flatten(0, 1).float(), labels.flatten(0, 1)
        )

    # init_device = device_type
    # buffer_device = None

    # model = model.to(device_type)
    # apply PT-D Tensor Parallel, activation checkpointing, torch.compile, Data Parallel
    parallelize_llama(
        model,
        world_mesh,
        parallel_dims,
        activation_checkpoint=task_config.activation_checkpoint,
    )
    with training.set_default_dtype(torch.bfloat16), device:
        for m in model.modules():
            # RoPE is not covered in state dict
            if hasattr(m, "rope_init"):
                m.rope_init()

    # model.to_empty(device=init_device)
    # with torch.no_grad():
    #     model.init_weights(buffer_device=buffer_device)
    with torch.no_grad():
        ckpt_path = PRETRAINED_MODEL_CKPT_PATH_MAPS[task_config.model_name_or_path]
        state_dict = load_checkpoint(ckpt_path=ckpt_path, model_name=task_config.model_name_or_path)
        is_rank_0 = torch.distributed.get_rank() == 0
        training.load_from_full_model_state_dict(
            model=model,
            full_sd=state_dict,
            device=device_type,
            is_rank_zero=is_rank_0,
            strict=True,
        )


    model.train()

    model_parts = [model]

    device_mem_stats = device_memory_monitor.get_peak_stats()
    logger.info(
        f"{device_type.upper()} memory usage for model: "
        f"{device_mem_stats.max_reserved_gib:.2f}GiB"
        f"({device_mem_stats.max_reserved_pct:.2f}%)"
    )

    # build optimizer after applying parallelisms to the model
    optimizers = build_optimizers(
        model_parts,
        lr=task_config.training_recipe.lr,
        fused=task_config.training_recipe.fused,
    )
    lr_schedulers = build_lr_schedulers(
        optimizers.optimizers,
        steps=task_config.training_recipe.max_steps,
        warmup_steps=task_config.training_recipe.warmup_steps,
        scheduler_type=scheduler_type,
    )

    train_state = TrainState()

    # load initial checkpoint
    checkpoint = CheckpointManager(
        dataloader=data_loader,
        model_parts=model_parts,
        optimizers=optimizers,
        lr_schedulers=lr_schedulers,
        states={"train_state": train_state},
        ckpt_config=task_config.ckpt_config,
        job_dump_folder=job_dump_folder,
    )

    if task_config.ckpt_config.create_seed_checkpoint:
        assert (
            world_size == 1
        ), "Must create seed-checkpoint using one gpu, to disable sharding"
        checkpoint.save(curr_step=0, force=True)
        logger.info("Created seed checkpoint")
        return

    checkpoint.load(step=task_config.ckpt_config.load_step)
    if use_wandb_for_log:
        enable_wandb = True
        enable_tensorboard = False
    else:
        enable_wandb = False
        enable_tensorboard = True
    metric_logger = build_metric_logger(
        parallel_dims,
        dump_folder=job_dump_folder,
        enable_tensorboard=enable_tensorboard,
        enable_wandb=enable_wandb,
        wandb_name=config_name,
    )

    # plot losses loaded from checkpoint (if any) to TensorBoard
    # NOTE: Loss info after the last log step before checkpoint saving will not be ploted.
    #       This can be avoided by setting checkpoint.interval to be a multiple of metrics.log_freq
    if train_state.step > 0:
        for idx, step in enumerate(train_state.log_steps):
            metrics = {
                "loss_metrics/global_avg_loss": train_state.global_avg_losses[idx],
                "loss_metrics/global_max_loss": train_state.global_max_losses[idx],
            }
            metric_logger.log(metrics, step=step)

    data_iterator = iter(data_loader)

    # variables used to keep info for metrics logging
    losses_since_last_log = []
    ntokens_since_last_log = 0
    data_loading_times = []
    time_last_log = time.perf_counter()
    device_memory_monitor.reset_peak_stats()

    checkpoint.reset()

    # train loop
    logger.info(
        f"Training starts at step {train_state.step + 1}, "
        f"with local batch size {local_batch_size}, "
        f"global batch size {task_config.training_recipe.batch_size}, "
        f"sequence length {task_config.seq_len}, "
        f"total steps {task_config.training_recipe.max_steps} "
        f"(warmup {task_config.training_recipe.warmup_steps})"
    )
    with maybe_enable_profiling(
        # TODO: check if this works
        # KVLinkDeveloper: disable profiling to save disk space.
        enable_profiling=False,
        job_dump_folder=job_dump_folder,
        global_step=train_state.step
    ) as torch_profiler, maybe_enable_memory_snapshot(
        global_step=train_state.step
    ) as memory_profiler:
        while train_state.step < task_config.training_recipe.max_steps:
            train_state.step += 1
            gc_handler.run(train_state.step)

            # get batch
            data_load_start = time.perf_counter()
            batch = next(data_iterator)
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            segment_ids = batch["segment_ids"]
            attention_mask = make_segment_mask(
                source_segments=segment_ids,
                target_segments=segment_ids,
                add_causal_lm_mask=True,
            )

            ntokens_since_last_log += labels.numel()
            data_loading_times.append(time.perf_counter() - data_load_start)

            input_ids = input_ids.to(device_type)
            labels = labels.to(device_type)
            attention_mask = attention_mask.to(device_type)
            optimizers.zero_grad()

            # Non-PP forward / backward
            pred = model(tokens=input_ids, mask=attention_mask)
            if isinstance(pred, list):
                # get the output logits
                pred = pred[-1]
            pred = pred[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()
            loss = loss_fn(pred, labels)
            del pred
            loss.backward()

            # clip gradients
            grad_norm = utils.clip_grad_norm_(
                [p for m in model_parts for p in m.parameters()],
                task_config.training_recipe.max_norm,
                foreach=True,
                pp_mesh=pp_mesh if parallel_dims.pp_enabled else None,
            )

            # optimizer step
            checkpoint.maybe_wait_for_staging()
            optimizers.step()
            lr_schedulers.step()

            losses_since_last_log.append(loss)

            # log metrics
            if (
                train_state.step == 1
                or train_state.step % log_freq == 0
            ):
                losses = [loss.item() for loss in losses_since_last_log]
                avg_loss, max_loss = sum(losses) / len(losses), max(losses)
                if (
                    parallel_dims.dp_replicate_enabled
                    or parallel_dims.dp_shard_enabled
                    or parallel_dims.cp_enabled
                ):
                    global_avg_loss, global_max_loss = (
                        utils.dist_mean(avg_loss, world_mesh["dp_cp"]),
                        utils.dist_max(max_loss, world_mesh["dp_cp"]),
                    )
                    grad_norm = grad_norm.item()
                    global_grad_norm = utils.dist_mean(grad_norm, world_mesh["dp_cp"])
                else:
                    global_avg_loss, global_max_loss = avg_loss, max_loss

                # update train state
                train_state.log_steps.append(train_state.step)
                train_state.global_avg_losses.append(global_avg_loss)
                train_state.global_max_losses.append(global_max_loss)

                time_delta = time.perf_counter() - time_last_log

                # tokens per second per device, abbreviated as tps
                tps = ntokens_since_last_log / (
                    time_delta * parallel_dims.non_data_parallel_size
                )
                # model FLOPS utilization
                # For its definition and calculation, please refer to the PaLM paper:
                # https://arxiv.org/abs/2204.02311

                time_end_to_end = time_delta / log_freq
                time_data_loading = sum(data_loading_times) / len(data_loading_times)
                time_data_loading_pct = 100 * sum(data_loading_times) / time_delta

                device_mem_stats = device_memory_monitor.get_peak_stats()

                curr_lr = optimizers.optimizers[0].param_groups[0]["lr"]

                metrics = {
                    "loss_metrics/global_avg_loss": global_avg_loss,
                    "loss_metrics/global_max_loss": global_max_loss,
                    "throughput(tps)": tps,
                    "time_metrics/end_to_end(s)": time_end_to_end,
                    "time_metrics/data_loading(s)": time_data_loading,
                    "time_metrics/data_loading(%)": time_data_loading_pct,
                    "memory/max_active(GiB)": device_mem_stats.max_active_gib,
                    "memory/max_active(%)": device_mem_stats.max_active_pct,
                    "memory/max_reserved(GiB)": device_mem_stats.max_reserved_gib,
                    "memory/max_reserved(%)": device_mem_stats.max_reserved_pct,
                    "memory/num_alloc_retries": device_mem_stats.num_alloc_retries,
                    "memory/num_ooms": device_mem_stats.num_ooms,
                    "training/lr": curr_lr,
                    "training/grad_norm": global_grad_norm,
                }
                metric_logger.log(metrics, step=train_state.step)

                logger.info(
                    f"{color.cyan}step: {train_state.step:2}  "
                    f"{color.green}loss: {global_avg_loss:7.4f}  "
                    f"{color.yellow}memory: {device_mem_stats.max_reserved_gib:5.2f}GiB"
                    f"({device_mem_stats.max_reserved_pct:.2f}%)  "
                    f"{color.blue}tps: {round(tps):,}  "
                    f"{color.magenta}%{color.reset}"
                )

                losses_since_last_log.clear()
                ntokens_since_last_log = 0
                data_loading_times.clear()
                time_last_log = time.perf_counter()
                device_memory_monitor.reset_peak_stats()

            checkpoint.save(
                train_state.step, force=(train_state.step == task_config.training_recipe.max_steps)
            )

            ###############################################################################
            # NEW EVALUATION CODE TO ADD (e.g., right before the end of the training loop)
            ###############################################################################
            if train_state.step % eval_interval == 0:
                model.eval()
                all_eval_losses = {}

                is_rank_0 = (torch.distributed.get_rank() == 0)

                with torch.no_grad():
                    for eval_dataset_name, eval_loader in eval_data_loader_dict.items():
                        eval_losses = []
                        if is_rank_0:
                            eval_iterator = tqdm.tqdm(eval_loader, desc=f"Eval {eval_dataset_name}")
                        else:
                            eval_iterator = eval_loader
                        for eval_batch in eval_iterator:
                            # move input to device
                            input_ids = eval_batch["input_ids"].to(device_type)
                            labels = eval_batch["labels"].to(device_type)
                            segment_ids = eval_batch["segment_ids"]
                            attention_mask = make_segment_mask(
                                source_segments=segment_ids,
                                target_segments=segment_ids,
                                add_causal_lm_mask=True,
                            )
                            # forward pass
                            pred = model(tokens=input_ids, mask=attention_mask)
                            if isinstance(pred, list):
                                pred = pred[-1]

                            # shift for causal LM cross-entropy
                            pred = pred[..., :-1, :].contiguous()
                            labels_ = labels[..., 1:].contiguous()

                            # compute loss
                            loss = loss_fn(pred, labels_)
                            eval_losses.append(loss.item())

                        # compute local average loss (on this rank)
                        local_avg_loss = sum(eval_losses) / len(eval_losses)

                        # reduce over DP/CP groups if necessary (to get a global average)
                        if parallel_dims.dp_enabled or parallel_dims.cp_enabled:
                            global_avg_loss = utils.dist_mean(local_avg_loss, world_mesh["dp_cp"])
                        else:
                            global_avg_loss = local_avg_loss

                        # store for logging
                        all_eval_losses[eval_dataset_name] = global_avg_loss

                # log and print eval metrics
                for eval_dataset_name, eval_loss in all_eval_losses.items():
                    metric_logger.log({f"eval/{eval_dataset_name}_loss": eval_loss}, step=train_state.step)
                    logger.info(
                        f"[Evaluation - {eval_dataset_name}] step: {train_state.step}  loss: {eval_loss:7.4f}"
                    )

                # restore train mode
                model.train()
            ###############################################################################


            # signal the profiler that the next profiling step has started
            if torch_profiler:
                torch_profiler.step()
            if memory_profiler:
                memory_profiler.step()

            # reduce timeout after first train step for faster signal
            # (assuming lazy init and compilation are finished)
            if train_state.step == 1:
                utils.set_pg_timeouts(
                    timeout=timedelta(seconds=train_timeout_seconds),
                    world_mesh=world_mesh,
                )

    if torch.distributed.get_rank() == 0:
        logger.info("Sleeping 2 seconds for other ranks to complete")
        time.sleep(2)

    metric_logger.close()
    logger.info("Training completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_name",
        default="block_datav1_step10k_bsz64_single_node",
        type=str,
    )
    parser.add_argument(
        "--use_wandb_for_log",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()
    config_name = args.config_name
    use_wandb_for_log = args.use_wandb_for_log
    main(config_name, use_wandb_for_log)
    torch.distributed.destroy_process_group()

