#!/usr/bin/env python3
from habitat_extensions.task import BertInstructionSensor
from vlnce_baselines.common.candidate_actions import register_candidate_actions
register_candidate_actions()
import argparse
import os
import random
import torch.distributed as dist
import numpy as np
import torch
from habitat import logger
from habitat_baselines.common.baseline_registry import baseline_registry
import vlnce_baselines.models.candidate_policy
import habitat_extensions  # noqa: F401
import vlnce_baselines  # noqa: F401
from vlnce_baselines.config.default import get_config
from vlnce_baselines.nonlearning_agents import (
    evaluate_agent,
    nonlearning_inference,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-type",
        choices=["train", "eval", "inference"],
        required=True,
        help="run type of the experiment (train, eval, inference)",
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )

    args = parser.parse_args()
    run_exp(**vars(args))


def run_exp(exp_config: str, run_type: str, opts=None) -> None:
    """Runs experiment given mode and config

    Args:
        exp_config: path to config file.
        run_type: "train" or "eval.
        opts: list of strings of additional config options.
    """
    config = get_config(exp_config, opts)
    # =========================================================================
    # 🔥🔥🔥 核心修改：DDP 分布式初始化逻辑 🔥🔥🔥
    # =========================================================================
    # torchrun 会自动设置 LOCAL_RANK 环境变量
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        
        # 1. 动态修改配置，把 GPU ID 改为当前进程的 Rank
        config.defrost()
        config.TORCH_GPU_ID = local_rank
        config.SIMULATOR_GPU_ID = local_rank
        config.freeze()

        # 2. 设置当前进程可见的 GPU
        torch.cuda.set_device(local_rank)
        
        # 3. 初始化进程组 (DDP 必须步骤)
        # 只有初始化了，后续 Trainer 里的 DDPModel 才能正常工作
        dist.init_process_group(backend="nccl", init_method="env://")
        
        logger.info(f"Process {local_rank} initialized DDP on GPU {local_rank}")
    # =========================================================================

    logger.info(f"config: {config}")
    logdir = "/".join(config.LOG_FILE.split("/")[:-1])
    if logdir:
        os.makedirs(logdir, exist_ok=True)
    logger.add_filehandler(config.LOG_FILE)

    random.seed(config.TASK_CONFIG.SEED)
    np.random.seed(config.TASK_CONFIG.SEED)
    torch.manual_seed(config.TASK_CONFIG.SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    if torch.cuda.is_available():
        torch.set_num_threads(1)

    if run_type == "eval":
        torch.backends.cudnn.deterministic = True
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # 清理一下显存cache
        if config.EVAL.EVAL_NONLEARNING:
            evaluate_agent(config)
            return

    if run_type == "inference" and config.INFERENCE.INFERENCE_NONLEARNING:
        nonlearning_inference(config)
        return

    trainer_init = baseline_registry.get_trainer(config.TRAINER_NAME)
    assert trainer_init is not None, f"{config.TRAINER_NAME} is not supported"
    trainer = trainer_init(config)

    if run_type == "train":
        trainer.train()
    elif run_type == "eval":
        trainer.eval()
    elif run_type == "inference":
        trainer.inference()


if __name__ == "__main__":
    main()
