import gc
import os
import random
import warnings
from collections import defaultdict
import torch.distributed as dist
import lmdb
import msgpack_numpy
import numpy as np
import torch
import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from habitat import logger
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
)
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.utils.common import batch_obs

from vlnce_baselines.common.aux_losses import AuxLosses
from vlnce_baselines.common.base_il_trainer import BaseVLNCETrainer
from vlnce_baselines.common.env_utils import construct_envs
from vlnce_baselines.common.utils import extract_instruction_tokens

from torch.cuda.amp import autocast, GradScaler # 🔥 新增导入
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    # import tensorflow as tf  # noqa: F401


class ObservationsDict(dict):
    def pin_memory(self):
        for k, v in self.items():
            self[k] = v.pin_memory()

        return self


def collate_fn(batch):
    """Each sample in batch: (
        obs,
        prev_actions,
        oracle_actions,
        inflec_weight,
    )
    """

    def _pad_helper(t, max_len, fill_val=0):
        pad_amount = max_len - t.size(0)
        if pad_amount == 0:
            return t

        pad = torch.full_like(t[0:1], fill_val).expand(
            pad_amount, *t.size()[1:]
        )
        return torch.cat([t, pad], dim=0)

    transposed = list(zip(*batch))

    observations_batch = list(transposed[0])
    prev_actions_batch = list(transposed[1])
    corrected_actions_batch = list(transposed[2])
    weights_batch = list(transposed[3])
    B = len(prev_actions_batch)

    new_observations_batch = defaultdict(list)
    for sensor in observations_batch[0]:
        for bid in range(B):
            new_observations_batch[sensor].append(
                observations_batch[bid][sensor]
            )

    observations_batch = new_observations_batch

    max_traj_len = max(ele.size(0) for ele in prev_actions_batch)
    for bid in range(B):
        for sensor in observations_batch:
            observations_batch[sensor][bid] = _pad_helper(
                observations_batch[sensor][bid], max_traj_len, fill_val=1.0
            )

        prev_actions_batch[bid] = _pad_helper(
            prev_actions_batch[bid], max_traj_len
        )
        corrected_actions_batch[bid] = _pad_helper(
            corrected_actions_batch[bid], max_traj_len
        )
        weights_batch[bid] = _pad_helper(weights_batch[bid], max_traj_len)

    for sensor in observations_batch:
        observations_batch[sensor] = torch.stack(
            observations_batch[sensor], dim=1
        )
        observations_batch[sensor] = observations_batch[sensor].view(
            -1, *observations_batch[sensor].size()[2:]
        )

    prev_actions_batch = torch.stack(prev_actions_batch, dim=1)
    corrected_actions_batch = torch.stack(corrected_actions_batch, dim=1)
    weights_batch = torch.stack(weights_batch, dim=1)
    not_done_masks = torch.ones_like(
        corrected_actions_batch, dtype=torch.uint8
    )
    not_done_masks[0] = 0

    observations_batch = ObservationsDict(observations_batch)

    return (
        observations_batch,
        prev_actions_batch.view(-1, 1),
        not_done_masks.view(-1, 1),
        corrected_actions_batch,
        weights_batch,
    )


def _block_shuffle(lst, block_size):
    blocks = [lst[i : i + block_size] for i in range(0, len(lst), block_size)]
    random.shuffle(blocks)

    return [ele for block in blocks for ele in block]


class IWTrajectoryDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        lmdb_features_dir,
        use_iw,
        inflection_weight_coef=1.0,
        lmdb_map_size=1e9,
        batch_size=1,
    ):
        super().__init__()
        self.lmdb_features_dir = lmdb_features_dir
        self.lmdb_map_size = lmdb_map_size
        self.preload_size = batch_size * 100
        self._preload = []
        self.batch_size = batch_size

        if use_iw:
            self.inflec_weights = torch.tensor([1.0, inflection_weight_coef])
        else:
            self.inflec_weights = torch.tensor([1.0, 1.0])

        with lmdb.open(
            self.lmdb_features_dir,
            map_size=int(self.lmdb_map_size),
            readonly=True,
            lock=False,
        ) as lmdb_env:
            self.length = lmdb_env.stat()["entries"]

    def _load_next(self):
        if len(self._preload) == 0:
            if len(self.load_ordering) == 0:
                raise StopIteration

            new_preload = []
            lengths = []
            with lmdb.open(
                self.lmdb_features_dir,
                map_size=int(self.lmdb_map_size),
                readonly=True,
                lock=False,
            ) as lmdb_env, lmdb_env.begin(buffers=True) as txn:
                for _ in range(self.preload_size):
                    if len(self.load_ordering) == 0:
                        break

                    new_preload.append(
                        msgpack_numpy.unpackb(
                            txn.get(str(self.load_ordering.pop()).encode()),
                            raw=False,
                        )
                    )

                    lengths.append(len(new_preload[-1][0]))

            sort_priority = list(range(len(lengths)))
            random.shuffle(sort_priority)

            sorted_ordering = list(range(len(lengths)))
            sorted_ordering.sort(key=lambda k: (lengths[k], sort_priority[k]))

            for idx in _block_shuffle(sorted_ordering, self.batch_size):
                self._preload.append(new_preload[idx])

        return self._preload.pop()

    def __next__(self):
        obs, prev_actions, oracle_actions = self._load_next()

        for k, v in obs.items():
            obs[k] = torch.from_numpy(np.copy(v))

        prev_actions = torch.from_numpy(np.copy(prev_actions))
        oracle_actions = torch.from_numpy(np.copy(oracle_actions))

        inflections = torch.cat(
            [
                torch.tensor([1], dtype=torch.long),
                (oracle_actions[1:] != oracle_actions[:-1]).long(),
            ]
        )

        return (
            obs,
            prev_actions,
            oracle_actions,
            self.inflec_weights[inflections],
        )
    # 下面是原来的代码
    # def __iter__(self):
    #     worker_info = torch.utils.data.get_worker_info()
    #     if worker_info is None:
    #         start = 0
    #         end = self.length
    #     else:
    #         per_worker = int(np.ceil(self.length / worker_info.num_workers))

    #         start = per_worker * worker_info.id
    #         end = min(start + per_worker, self.length)

    #     # Reverse so we can use .pop()
    #     self.load_ordering = list(
    #         reversed(
    #             _block_shuffle(list(range(start, end)), self.preload_size)
    #         )
    #     )

    #     return self
    # 原来的代码结束

    # 下面是修改后的代码
    def __iter__(self):
        # 1. 获取分布式信息 (DDP)
        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1

        # 2. 获取 worker 信息 (DataLoader num_workers)
        worker_info = torch.utils.data.get_worker_info()
        
        # 3. 计算这一块 GPU 应该负责的总区间 (GPU Sharding)
        # 将整个数据集平均分成 world_size 份
        per_gpu_length = int(np.ceil(self.length / world_size))
        gpu_start = rank * per_gpu_length
        gpu_end = min(gpu_start + per_gpu_length, self.length)

        # 4. 在这块 GPU 的区间内，再分配给不同的 CPU worker (Worker Sharding)
        if worker_info is None:
            # 单进程读取
            start = gpu_start
            end = gpu_end
        else:
            # 多进程读取：计算当前 GPU 区间内的切片
            # 这里的逻辑是：在 gpu_start 到 gpu_end 的范围内再切分
            valid_length = gpu_end - gpu_start
            per_worker = int(np.ceil(valid_length / worker_info.num_workers))
            
            worker_id = worker_info.id
            start = gpu_start + worker_id * per_worker
            end = min(start + per_worker, gpu_end)

        # 5. 生成加载顺序
        # Reverse so we can use .pop()
        # 注意：这里 range 的范围已经是切分好的 [start, end)
        if start >= end:
            self.load_ordering = [] # 这个 worker/rank 不需要干活
        else:
            self.load_ordering = list(
                reversed(
                    _block_shuffle(list(range(start, end)), self.preload_size)
                )
            )

        return self
    # 修改结束

@baseline_registry.register_trainer(name="dagger")
class DaggerTrainer(BaseVLNCETrainer):
    def __init__(self, config=None):
        self.lmdb_features_dir = config.IL.DAGGER.lmdb_features_dir.format(
            split=config.TASK_CONFIG.DATASET.SPLIT
        )
        super().__init__(config)

        # 🔥 新增：初始化梯度缩放器（用于混合精度）
        self.scaler = GradScaler()
    def _update_agent(
        self,
        observations,
        prev_actions,
        not_done_masks,
        corrected_actions,
        weights,
        step_grad: bool = True,
        loss_accumulation_scalar: int = 1,
    ):
        T, N = corrected_actions.size()

        # 自动判断是否使用了 DDP
        net = self.policy.net.module if hasattr(self.policy.net, "module") else self.policy.net

        recurrent_hidden_states = torch.zeros(
            N,
            net.num_recurrent_layers,
            self.config.MODEL.STATE_ENCODER.hidden_size,
            device=self.device,
        )

        AuxLosses.clear()

        # 🔥 1. 开启前向传播的自动混合精度
        with autocast():
            distribution = self.policy.build_distribution(
                observations, recurrent_hidden_states, prev_actions, not_done_masks
            )

            logits = distribution.logits
            logits = logits.view(T, N, -1)

            # 交叉熵计算 (在 autocast 下会自动处理为稳定精度)
            action_loss = F.cross_entropy(
                logits.permute(0, 2, 1), corrected_actions, reduction="none"
            )
            action_loss = ((weights * action_loss).sum(0) / weights.sum(0)).mean()

            aux_mask = (weights > 0).view(-1)
            aux_loss = AuxLosses.reduce(aux_mask)

            loss = action_loss + aux_loss
            loss = loss / loss_accumulation_scalar

        # 🔥 2. 使用 scaler 缩放损失并进行反向传播
        # 代替原来的 loss.backward()
        self.scaler.scale(loss).backward()

        if step_grad:
            # 如果你有梯度裁剪，在这里添加：
            # self.scaler.unscale_(self.optimizer)
            # torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm)

            # 🔥 3. 使用 scaler.step 更新参数并更新 scaler 状态
            # 代替原来的 self.optimizer.step()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            self.optimizer.zero_grad()

        if isinstance(aux_loss, torch.Tensor):
            aux_loss = aux_loss.item()
            
        return loss.item(), action_loss.item(), aux_loss
    # ------------------ 🔥 新增修复代码开始 🔥 ------------------
    def load_checkpoint(self, checkpoint_path, *args, **kwargs):
        """
        覆盖父类的 load_checkpoint 方法。
        主要目的是在加载权重前，自动去除 'module.' 前缀，
        解决从 DDP 多卡训练保存的模型加载到单卡或其他环境时的 Key Mismatch 问题。
        """
        # 1. 检查文件是否存在
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        # 2. 使用 torch.load 加载
        # map_location 确保加载到当前设备
        ckpt = torch.load(checkpoint_path, map_location=self.device)

        # 3. 清洗 state_dict 中的 'module.' 前缀
        if "state_dict" in ckpt:
            from collections import OrderedDict
            state_dict = ckpt["state_dict"]
            new_state_dict = OrderedDict()
            
            fixed_count = 0
            for k, v in state_dict.items():
                if k.startswith("module."):
                    name = k.replace("module.", "")
                    fixed_count += 1
                else:
                    name = k
                new_state_dict[name] = v
            
            # 将清洗后的字典放回 ckpt
            ckpt["state_dict"] = new_state_dict
            
            # 打印日志（防止所有进程都打印，只让主进程打印）
            if not dist.is_initialized() or dist.get_rank() == 0:
                if fixed_count > 0:
                    logger.info(f"✅ Fixed {fixed_count} keys by removing 'module.' prefix from {checkpoint_path}")
                else:
                    logger.info(f"Loaded checkpoint from {checkpoint_path} (No prefix fix needed).")

        return ckpt
    # ------------------ 🔥 新增修复代码结束 🔥 ------------------

    def _make_dirs(self) -> None:
        self._make_ckpt_dir()
        os.makedirs(self.lmdb_features_dir, exist_ok=True)
        if self.config.EVAL.SAVE_RESULTS:
            self._make_results_dir()

    def _update_dataset(self, data_it):
        if torch.cuda.is_available():
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()

        envs = construct_envs(self.config, get_env_class(self.config.ENV_NAME))
        expert_uuid = self.config.IL.DAGGER.expert_policy_sensor_uuid
        
        # 🔥🔥🔥【修复开始】处理 DDP 模型解包 🔥🔥🔥
        # 获取底层的模型 (unwrapped)，以便访问 num_recurrent_layers 等属性
        # 如果 self.policy.net 是 DDP 对象，取 .module；否则直接用
        net_module = self.policy.net
        if hasattr(net_module, "module"):
            net_module = net_module.module
        # 🔥🔥🔥【修复结束】🔥🔥🔥


        rnn_states = torch.zeros(
            envs.num_envs,
            net_module.num_recurrent_layers, # 👈 修改这里：使用 net_module
            # self.policy.net.num_recurrent_layers,
            self.config.MODEL.STATE_ENCODER.hidden_size,
            device=self.device,
        )
        prev_actions = torch.zeros(
            envs.num_envs,
            1,
            device=self.device,
            dtype=torch.long,
        )
        not_done_masks = torch.zeros(
            envs.num_envs, 1, dtype=torch.uint8, device=self.device
        )

        observations = envs.reset()
        observations = extract_instruction_tokens(
            observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID
        )
        batch = batch_obs(observations, self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        episodes = [[] for _ in range(envs.num_envs)]
        skips = [False for _ in range(envs.num_envs)]
        # Populate dones with False initially
        dones = [False for _ in range(envs.num_envs)]

        # https://arxiv.org/pdf/1011.0686.pdf
        # Theoretically, any beta function is fine so long as it converges to
        # zero as data_it -> inf. The paper suggests starting with beta = 1 and
        # exponential decay.
        p = self.config.IL.DAGGER.p
        # in Python 0.0 ** 0.0 == 1.0, but we want 0.0
        beta = 0.0 if p == 0.0 else p ** data_it

        ensure_unique_episodes = beta == 1.0

        def hook_builder(tgt_tensor):
            def hook(m, i, o):
                tgt_tensor.set_(o.cpu())

            return hook

        rgb_features = None
        rgb_hook = None
        if not self.config.MODEL.RGB_ENCODER.trainable:
            rgb_features = torch.zeros((1,), device="cpu")
            # rgb_hook = self.policy.net.rgb_encoder.cnn.register_forward_hook(
            #     hook_builder(rgb_features)
            # )
            # 👈 修改这里：使用 net_module 防止 DDP 下找不到 encoder
            rgb_hook = net_module.rgb_encoder.cnn.register_forward_hook(
                hook_builder(rgb_features)
            )

        depth_features = None
        depth_hook = None
        if not self.config.MODEL.DEPTH_ENCODER.trainable:
            depth_features = torch.zeros((1,), device="cpu")
            # depth_hook = self.policy.net.depth_encoder.visual_encoder.register_forward_hook(
            #     hook_builder(depth_features)
            # )
            # 👈 修改这里：使用 net_module 防止 DDP 下找不到 encoder
            depth_hook = net_module.depth_encoder.visual_encoder.register_forward_hook(
                hook_builder(depth_features)
            )

        collected_eps = 0
        ep_ids_collected = None
        if ensure_unique_episodes:
            ep_ids_collected = {
                ep.episode_id for ep in envs.current_episodes()
            }

        with tqdm.tqdm(
            total=self.config.IL.DAGGER.update_size, dynamic_ncols=True
        ) as pbar, lmdb.open(
            self.lmdb_features_dir,
            map_size=int(self.config.IL.DAGGER.lmdb_map_size),
        ) as lmdb_env, torch.no_grad():
            
            # # ✅ 1. 计算还需要跑多少条
            # start_id = lmdb_env.stat()["entries"]
            # # 原始代码：collected_eps = 0
            # # 修改后：我们只收集“剩下”的部分
            # target_size = self.config.IL.DAGGER.update_size # 目标总数 (157232)
            # needed_eps = target_size - start_id # 还需要跑多少 (例如剩下 10000)
            # # 如果已经收集够了，直接返回
            # if needed_eps <= 0:
            #     logger.info("Data collection complete. Skipping.")
            #     envs.close()
            #     return

            # 1. 获取当前数据库里已经有的条数
            start_id = lmdb_env.stat()["entries"]

            # 2. 【核心修改】计算这一轮结束时，数据库应该有的总条数
            # data_it 是从 0 开始的 (Iter 0, Iter 1...)
            # Iter 0 结束应有 1 * update_size
            # Iter 1 结束应有 2 * update_size
            per_iter_size = self.config.IL.DAGGER.update_size
            target_cumulative_size = per_iter_size * (data_it + 1)

            # 3. 计算缺口：还需要补多少条？
            needed_eps = target_cumulative_size - start_id
            
            # 打印日志方便调试
            info_msg = f"DAgger Iter: {data_it} | Existing: {start_id} | Target Total: {target_cumulative_size} | Needed: {needed_eps}"
            if dist.is_initialized():
                if dist.get_rank() == 0: logger.info(info_msg)
            else:
                logger.info(info_msg)

            # 4. 如果缺口 <= 0，说明这一轮的数据以前已经跑过了（断点续传），直接跳过
            if needed_eps <= 0:
                logger.info("Data already sufficient for this iteration. Skipping collection.")
                envs.close()
                # 记得移除 hook 防止内存泄漏
                if rgb_hook is not None: rgb_hook.remove()
                if depth_hook is not None: depth_hook.remove()
                return

            # 5. 重置进度条，只显示本次需要采集的数量
            pbar.reset(total=needed_eps)
            # pbar.update(start_id) # 更新进度条到当前位置
            txn = lmdb_env.begin(write=True)
            # ✅ 这里的 collected_eps 必须从 0 开始！
            # 因为下面的 txn.put 用的是 start_id + collected_eps
            collected_eps = 0 

            # ✅ 2. 修改循环条件，只跑剩下的量
            while collected_eps < needed_eps:
            # 修改结束

            # 下面三行是官方老代码
            # start_id = lmdb_env.stat()["entries"]
            # txn = lmdb_env.begin(write=True)

            # while collected_eps < self.config.IL.DAGGER.update_size:
                current_episodes = None
                envs_to_pause = None
                if ensure_unique_episodes:
                    envs_to_pause = []
                    current_episodes = envs.current_episodes()

                for i in range(envs.num_envs):
                    if dones[i] and not skips[i]:
                        ep = episodes[i]
                        traj_obs = batch_obs(
                            [step[0] for step in ep],
                            device=torch.device("cpu"),
                        )
                        del traj_obs[expert_uuid]
                        for k, v in traj_obs.items():
                            traj_obs[k] = v.numpy()
                            if self.config.IL.DAGGER.lmdb_fp16:
                                traj_obs[k] = traj_obs[k].astype(np.float16)

                        transposed_ep = [
                            traj_obs,
                            np.array([step[1] for step in ep], dtype=np.int64),
                            np.array([step[2] for step in ep], dtype=np.int64),
                        ]
                        txn.put(
                            str(start_id + collected_eps).encode(),
                            msgpack_numpy.packb(
                                transposed_ep, use_bin_type=True
                            ),
                        )

                        pbar.update()
                        collected_eps += 1

                        # ------------------ 新增代码开始 ------------------
                        # 每采集 50 个 episode 强制清理一次内存
                        # 这能有效防止内存碎片化导致的 OOM
                        if collected_eps % 5000 == 0:
                            gc.collect()
                        # ------------------ 新增代码结束 ------------------

                        if (
                            collected_eps
                            % self.config.IL.DAGGER.lmdb_commit_frequency
                        ) == 0:
                            txn.commit()
                            txn = lmdb_env.begin(write=True)

                        if ensure_unique_episodes:
                            if (
                                current_episodes[i].episode_id
                                in ep_ids_collected
                            ):
                                envs_to_pause.append(i)
                            else:
                                ep_ids_collected.add(
                                    current_episodes[i].episode_id
                                )

                    if dones[i]:
                        episodes[i] = []

                if ensure_unique_episodes:
                    (
                        envs,
                        rnn_states,
                        not_done_masks,
                        prev_actions,
                        batch,
                        _,
                    ) = self._pause_envs(
                        envs_to_pause,
                        envs,
                        rnn_states,
                        not_done_masks,
                        prev_actions,
                        batch,
                    )
                    if envs.num_envs == 0:
                        break

                actions, rnn_states = self.policy.act(
                    batch,
                    rnn_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=False,
                )
                actions = torch.where(
                    torch.rand_like(actions, dtype=torch.float) < beta,
                    batch[expert_uuid].long(),
                    actions,
                )

                for i in range(envs.num_envs):
                    if rgb_features is not None:
                        observations[i]["rgb_features"] = rgb_features[i]
                        del observations[i]["rgb"]

                    if depth_features is not None:
                        observations[i]["depth_features"] = depth_features[i]
                        del observations[i]["depth"]

                    episodes[i].append(
                        (
                            observations[i],
                            prev_actions[i].item(),
                            batch[expert_uuid][i].item(),
                        )
                    )

                skips = batch[expert_uuid].long() == -1
                actions = torch.where(
                    skips, torch.zeros_like(actions), actions
                )
                skips = skips.squeeze(-1).to(device="cpu", non_blocking=True)
                prev_actions.copy_(actions)

                outputs = envs.step([a[0].item() for a in actions])
                observations, _, dones, _ = [list(x) for x in zip(*outputs)]
                observations = extract_instruction_tokens(
                    observations,
                    self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
                )
                batch = batch_obs(observations, self.device)
                batch = apply_obs_transforms_batch(batch, self.obs_transforms)

                not_done_masks = torch.tensor(
                    [[0] if done else [1] for done in dones],
                    dtype=torch.uint8,
                    device=self.device,
                )

            txn.commit()

        envs.close()
        envs = None

        if rgb_hook is not None:
            rgb_hook.remove()
        if depth_hook is not None:
            depth_hook.remove()

    def train(self) -> None:
        """Main method for training DAgger."""
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        if self.config.IL.DAGGER.preload_lmdb_features:
            try:
                lmdb.open(self.lmdb_features_dir, readonly=True, lock=False)
            except lmdb.Error as err:
                logger.error(
                    "Cannot open database for teacher forcing preload."
                )
                raise err
        else:
            # 下面五行是官方老代码
            # with lmdb.open(
            #     self.lmdb_features_dir,
            #     map_size=int(self.config.IL.DAGGER.lmdb_map_size),
            # ) as lmdb_env, lmdb_env.begin(write=True) as txn:
            #     txn.drop(lmdb_env.open_db())

            # 下面是新的代码
            # ✅ 新增逻辑：只有当数据库为空或不存在时，才执行清空操作
            # ⚠️【修改开始】断点续传逻辑
            # 先以只读模式打开，看看里面有多少数据
            current_entries = 0
            # 1. 检查路径是否存在，并且检查里面是否有文件
            # 如果文件夹不存在，或者文件夹存在但为空列表（[]），都算作 0
            if os.path.exists(self.lmdb_features_dir) and len(os.listdir(self.lmdb_features_dir)) > 0:
                try:
                    with lmdb.open(self.lmdb_features_dir, readonly=True, lock=False) as lmdb_env:
                        current_entries = lmdb_env.stat()["entries"]
                except (lmdb.Error, Exception):
                    # 如果文件损坏或打不开，也重置为 0
                    logger.info("Existing LMDB is corrupted or empty. Starting from scratch.")
                    current_entries = 0
            # 只有当数据是 0 或者不存在时，才执行 drop (清空)
            # 否则我们认为是想接着跑
            if current_entries == 0:
                with lmdb.open(
                    self.lmdb_features_dir,
                    map_size=int(self.config.IL.DAGGER.lmdb_map_size),
                ) as lmdb_env, lmdb_env.begin(write=True) as txn:
                    txn.drop(lmdb_env.open_db())
                logger.info("Created new LMDB database.")
            else:
                logger.info(f"Found {current_entries} entries, resuming...")
            # ⚠️【修改结束】断点续传逻辑

        EPS = self.config.IL.DAGGER.expert_policy_sensor
        if EPS not in self.config.TASK_CONFIG.TASK.SENSORS:
            self.config.TASK_CONFIG.TASK.SENSORS.append(EPS)

        self.config.defrost()

        # if doing teacher forcing, don't switch the scene until it is complete
        if self.config.IL.DAGGER.p == 1.0:
            self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = (
                -1
            )
        self.config.freeze()

        observation_space, action_space = self._get_spaces(self.config)

        self._initialize_policy(
            self.config,
            self.config.IL.load_from_ckpt,
            observation_space=observation_space,
            action_space=action_space,
        )
        
        # 🔥🔥🔥【新增】DDP 模型包装：实现梯度的自动结合 🔥🔥🔥
        if dist.is_initialized():
            # 获取当前设备
            device_id = self.device
            # 包装模型
            # find_unused_parameters=True 是为了防止因为 CLIP 冻结参数导致的报错
            self.policy.net = DDP(
                self.policy.net, 
                device_ids=[device_id], 
                output_device=device_id, 
                find_unused_parameters=False
            )
            logger.info(f"Process {dist.get_rank()}: Wrapped model with DDP")
        # 🔥🔥🔥【新增】DDP 模型包装结束 🔥🔥🔥
        
        with TensorboardWriter(
            self.config.TENSORBOARD_DIR,
            flush_secs=self.flush_secs,
            purge_step=0,
        ) as writer:
            for dagger_it in range(self.config.IL.DAGGER.iterations):

                # ------------------ 🔥 新增修复代码开始 (逻辑控制) 🔥 ------------------
                # 计算当前这一轮 dagger_it 的起始 epoch
                # 假设每轮 DAgger 训练 4 个 epoch (self.config.IL.epochs = 4)
                # 这里的 self.start_epoch 是全局累积的 epoch (例如加载了 epoch 6)
                
                epochs_per_iter = self.config.IL.epochs
                
                # 1. 如果这一轮完全是以前跑过的 (例如当前 dagger_it=0, 但我们从 epoch 6 恢复)
                # 6 // 4 = 1，说明 Iter 0 已经跑完了
                if self.config.IL.load_from_ckpt and dagger_it < (self.start_epoch // epochs_per_iter):
                    if dist.get_rank() == 0:
                        logger.info(f"Skipping DAgger Iter {dagger_it} (Already trained in previous run).")
                    # 直接跳过这一轮，不采集数据，不加载 Dataset
                    continue

                # 2. 如果这一轮是“断点续传”的那一轮 (例如当前 dagger_it=1, 从 epoch 6 恢复)
                # 我们应该从第 2 个 epoch 开始跑 (6 % 4 = 2)
                elif self.config.IL.load_from_ckpt and dagger_it == (self.start_epoch // epochs_per_iter):
                    current_start_epoch = self.start_epoch % epochs_per_iter
                    if not dist.is_initialized() or dist.get_rank() == 0:
                        logger.info(f"Resuming DAgger Iter {dagger_it} from Epoch {current_start_epoch}.")
                
                # 3. 如果是全新的轮次 (例如 dagger_it=2)
                else:
                    current_start_epoch = 0
                # ------------------ 🔥 新增修复代码结束 🔥 ------------------    

                # ❌ 原代码是: step_id = 0
                step_id = self.step_id # 这里原来是0
                # 只有当这是全新的训练（非 Resume），且是第一轮时，才重置为 0
                if not self.config.IL.is_requeue and dagger_it == 0 and not self.config.IL.load_from_ckpt:
                    step_id = 0
                # 新增代码结束
                if not self.config.IL.DAGGER.preload_lmdb_features:
                    self._update_dataset(
                        dagger_it + (1 if self.config.IL.load_from_ckpt else 0)
                    )

                if torch.cuda.is_available():
                    with torch.cuda.device(self.device):
                        torch.cuda.empty_cache()
                gc.collect()

                dataset = IWTrajectoryDataset(
                    self.lmdb_features_dir,
                    self.config.IL.use_iw,
                    inflection_weight_coef=self.config.IL.inflection_weight_coef,
                    lmdb_map_size=self.config.IL.DAGGER.lmdb_map_size,
                    batch_size=self.config.IL.batch_size,
                )
                diter = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=self.config.IL.batch_size,
                    shuffle=False,
                    collate_fn=collate_fn,
                    pin_memory=False,
                    drop_last=True,  # drop last batch if smaller
                    num_workers=3, # 原先是3
                )

                AuxLosses.activate()
                for epoch in tqdm.trange(
                    current_start_epoch, self.config.IL.epochs, dynamic_ncols=True # 新增self.start_epoch
                ):
                    for batch in tqdm.tqdm(
                        diter,
                        total=(dataset.length // dataset.batch_size) // world_size,
                        leave=False,
                        dynamic_ncols=True,
                    ):
                        (
                            observations_batch,
                            prev_actions_batch,
                            not_done_masks,
                            corrected_actions_batch,
                            weights_batch,
                        ) = batch

                        observations_batch = {
                            k: v.to(
                                device=self.device,
                                dtype=torch.float32,
                                non_blocking=True,
                            )
                            for k, v in observations_batch.items()
                        }

                        loss, action_loss, aux_loss = self._update_agent(
                            observations_batch,
                            prev_actions_batch.to(
                                device=self.device, non_blocking=True
                            ),
                            not_done_masks.to(
                                device=self.device, non_blocking=True
                            ),
                            corrected_actions_batch.to(
                                device=self.device, non_blocking=True
                            ),
                            weights_batch.to(
                                device=self.device, non_blocking=True
                            ),
                        )
                        # 🔥🔥🔥【修改】只让 Rank 0 写日志 🔥🔥🔥
                        if dist.is_initialized():
                            rank = dist.get_rank()
                        else:
                            rank = 0

                        # 只有 rank 0 负责打印和写 TensorBoard
                        if rank == 0:
                            logger.info(f"train_loss: {loss}")
                            logger.info(f"train_action_loss: {action_loss}")
                            logger.info(f"train_aux_loss: {aux_loss}")
                            logger.info(f"Batches processed: {step_id}.")
                            logger.info(
                                f"On DAgger iter {dagger_it}, Epoch {epoch}."
                            )
                            writer.add_scalar(
                                f"train_loss_iter_{dagger_it}", loss, step_id
                            )
                            writer.add_scalar(
                                f"train_action_loss_iter_{dagger_it}",
                                action_loss,
                                step_id,
                            )
                            writer.add_scalar(
                                f"train_aux_loss_iter_{dagger_it}",
                                aux_loss,
                                step_id,
                            )

                        step_id += 1 
                        # ---------------------------------------------
                        # 下面是老代码
                        # logger.info(f"train_loss: {loss}")
                        # logger.info(f"train_action_loss: {action_loss}")
                        # logger.info(f"train_aux_loss: {aux_loss}")
                        # logger.info(f"Batches processed: {step_id}.")
                        # logger.info(
                        #     f"On DAgger iter {dagger_it}, Epoch {epoch}."
                        # )
                        # writer.add_scalar(
                        #     f"train_loss_iter_{dagger_it}", loss, step_id
                        # )
                        # writer.add_scalar(
                        #     f"train_action_loss_iter_{dagger_it}",
                        #     action_loss,
                        #     step_id,
                        # )
                        # writer.add_scalar(
                        #     f"train_aux_loss_iter_{dagger_it}",
                        #     aux_loss,
                        #     step_id,
                        # )
                        # step_id += 1  # noqa: SIM113
                        # 原来的代码结束
                        # ------------------ 新增代码 ------------------
                        if step_id % 2000 == 0:  # 每训练2000个batch清理一次
                            gc.collect()
                        # ---------------------------------------------

                    # 🔥🔥🔥【修改】只让 Rank 0 保存模型 🔥🔥🔥
                    if rank == 0:
                        self.save_checkpoint(
                            f"ckpt.{dagger_it * self.config.IL.epochs + epoch}.pth",
                            epoch=epoch,     # 传入当前 epoch
                            step_id=step_id  # 传入当前 step_id
                        )
                    # 下面是老代码
                    # self.save_checkpoint(
                    #     f"ckpt.{dagger_it * self.config.IL.epochs + epoch}.pth"
                    # )
                AuxLosses.deactivate()
