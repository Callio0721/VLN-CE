import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import habitat_sim

from gym import Space
from habitat import Config
from habitat.core.registry import registry
from habitat.core.embodied_task import SimulatorTaskAction
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.models.rnn_state_encoder import build_rnn_state_encoder
from habitat_baselines.rl.ppo.policy import Net, Policy
from vlnce_baselines.models.policy import ILPolicy
from vlnce_baselines.models.encoders.instruction_encoder import (
    InstructionEncoder,
    BertInstructionEncoder,
)
from vlnce_baselines.models.encoders import resnet_encoders

# =============================================================================
# 🔥 1. 核心工具函数：生成候选点 (带 Valid Flag)
# =============================================================================
def get_candidate_waypoints(sim, num_candidates=12, max_distance=2.5, min_distance=0.5):
    try:
        agent_state = sim.get_agent(0).get_state()
        agent_position = np.array(agent_state.position, dtype=np.float32)
        candidates = []
        angles = np.linspace(0, 2 * np.pi, num_candidates, endpoint=False)
        
        q = agent_state.rotation
        qx, qy, qz, qw = q.x, q.y, q.z, q.w
        
        for angle in angles:
            local_dir = np.array([np.sin(angle), 0, -np.cos(angle)], dtype=np.float32)
            
            # 手动旋转向量
            q_vec = np.array([qx, qy, qz], dtype=np.float32)
            uv = np.cross(q_vec, local_dir)
            uuv = np.cross(q_vec, uv)
            global_dir = local_dir + 2.0 * (qw * uv + uuv)
            
            target_pos = agent_position + global_dir * max_distance
            snapped_point = sim.pathfinder.snap_point(target_pos)
            
            is_valid = False
            dist = 0.0
            
            if not np.isnan(snapped_point).any():
                is_valid = sim.pathfinder.is_navigable(snapped_point)
                if is_valid:
                    dist = np.linalg.norm(snapped_point - agent_position)
                    if dist < min_distance:
                        is_valid = False

            # Features: [dist, sin, cos, valid]
            valid_flag = 1.0 if is_valid else 0.0
            
            # 即使点无效，我们也要把位置填进去（或者填当前位置），
            # 关键是 valid_flag 已经标记了它是坏的。
            # 这里为了防止 NaN，如果无效，位置设为 agent_position
            safe_pos = snapped_point if is_valid else agent_position

            feat = np.array([dist, np.sin(angle), np.cos(angle), valid_flag], dtype=np.float32)
            candidates.append({
                "position": safe_pos,
                "features": feat
            })
                
        return candidates

    except Exception as e:
        return [{"position": None, "features": np.zeros(4, dtype=np.float32)} for _ in range(num_candidates)]


# =============================================================================
# 🔥 2. 自定义动作注册区
# =============================================================================

class CandidateMoveActionBase(SimulatorTaskAction):
    def step(self, *args, **kwargs):
        try:
            cand_idx = int(self.name.split("_")[-1])
        except:
            cand_idx = 0

        # 获取观测的辅助函数，统一处理 RGBA -> RGB
        def get_safe_obs(sim):
            obs = sim.get_sensor_observations()
            if "rgb" in obs and obs["rgb"].shape[-1] == 4:
                obs["rgb"] = obs["rgb"][..., :3]
            return obs

        try:
            if self._sim.pathfinder is None:
                return get_safe_obs(self._sim)
            cands = get_candidate_waypoints(self._sim, num_candidates=12)
        except Exception as e:
            return get_safe_obs(self._sim)
        
        if cand_idx >= len(cands):
            return get_safe_obs(self._sim)

        cand = cands[cand_idx]
        # 注意：这里我们允许移动到看似无效的点（只要位置不为None）
        # 让物理引擎去处理碰撞，而不是在这里拦住
        if cand["position"] is not None:
            target_pos = cand["position"]
            agent = self._sim.get_agent(0)
            new_state = agent.get_state()
            new_state.position = target_pos
            agent.set_state(new_state)
            
        try:
            return get_safe_obs(self._sim)
        except Exception as e:
            raise e

# 批量注册
for i in range(12):
    action_name = f"CANDIDATE_MOVE_{i}"
    cls = type(f"CandidateMoveAction{i}", (CandidateMoveActionBase,), {"name": action_name})
    try:
        registry.register_task_action(cls, name=action_name)
    except (ValueError, AssertionError):
        pass

# =============================================================================
# 🔥 3. Policy 定义
# =============================================================================

@baseline_registry.register_policy
class CandidateCMAPolicy(ILPolicy):
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        model_config: Config,
    ) -> None:
        # 🚑 LAZY MONKEY PATCH
        try:
            from vlnce_baselines.dagger_trainer import DAggerTrainer
            def _patched_update_dataset(trainer_self, *args, **kwargs):
                if hasattr(trainer_self, 'prev_actions') and trainer_self.prev_actions.size(1) != 1:
                    new_prev_actions = torch.zeros(
                        trainer_self.prev_actions.size(0), 
                        1, 
                        dtype=trainer_self.prev_actions.dtype, 
                        device=trainer_self.prev_actions.device
                    )
                    trainer_self.prev_actions = new_prev_actions
                return DAggerTrainer._original_update_dataset(trainer_self, *args, **kwargs)

            if not hasattr(DAggerTrainer, "_original_update_dataset"):
                DAggerTrainer._original_update_dataset = DAggerTrainer._update_dataset
                DAggerTrainer._update_dataset = _patched_update_dataset
        except ImportError:
            pass

        num_candidates = model_config.CANDIDATE.num_candidates
        net = CandidateCMANet(
            observation_space=observation_space,
            model_config=model_config,
            num_candidates=num_candidates
        )
        super().__init__(
            net=net,
            dim_actions=num_candidates + 1 
        )
        self.num_candidates = num_candidates

    @classmethod
    def from_config(cls, config: Config, observation_space: Space, action_space: Space):
        return cls(
            observation_space=observation_space,
            action_space=action_space,
            model_config=config.MODEL,
        )

    def act(
        self,
        observations,
        rnn_states,
        prev_actions,
        masks,
        deterministic=False,
    ):
        logits, rnn_states_out = self.net(
            observations, rnn_states, prev_actions, masks
        )
        dist = torch.distributions.Categorical(logits=logits)
        
        if deterministic:
            action_indices = logits.argmax(dim=1)
        else:
            action_indices = dist.sample()

        if action_indices.dim() == 1:
            action_indices = action_indices.unsqueeze(-1)

        return action_indices, rnn_states_out

    def build_distribution(self, observations, rnn_states, prev_actions, masks):
        logits, _ = self.net(observations, rnn_states, prev_actions, masks)
        return torch.distributions.Categorical(logits=logits)


class CandidateCMANet(Net):
    def __init__(self, observation_space: Space, model_config: Config, num_candidates=12):
        super().__init__()
        self.model_config = model_config
        self.num_candidates = num_candidates
        
        if model_config.INSTRUCTION_ENCODER.rnn_type == "BERT":
            self.instruction_encoder = BertInstructionEncoder(model_config.INSTRUCTION_ENCODER)
        else:
            self.instruction_encoder = InstructionEncoder(model_config.INSTRUCTION_ENCODER)

        self.depth_encoder = getattr(resnet_encoders, model_config.DEPTH_ENCODER.cnn_type)(
            observation_space, output_size=model_config.DEPTH_ENCODER.output_size,
            checkpoint=model_config.DEPTH_ENCODER.ddppo_checkpoint,
            backbone=model_config.DEPTH_ENCODER.backbone, trainable=model_config.DEPTH_ENCODER.trainable,
            spatial_output=True,
        )
        self.rgb_encoder = getattr(resnet_encoders, model_config.RGB_ENCODER.cnn_type)(
            model_config.RGB_ENCODER.output_size, normalize_visual_inputs=model_config.normalize_rgb,
            trainable=model_config.RGB_ENCODER.trainable, spatial_output=True,
        )

        hidden_size = model_config.STATE_ENCODER.hidden_size
        self._hidden_size = hidden_size
        self.prev_action_embedding = nn.Embedding(num_candidates + 2, 32) 

        self.rgb_linear = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(self.rgb_encoder.output_shape[0], model_config.RGB_ENCODER.output_size), nn.ReLU(True))
        self.depth_linear = nn.Sequential(nn.Flatten(), nn.Linear(np.prod(self.depth_encoder.output_shape), model_config.DEPTH_ENCODER.output_size), nn.ReLU(True))

        rnn_input_size = model_config.DEPTH_ENCODER.output_size + model_config.RGB_ENCODER.output_size + 32
        self.state_encoder = build_rnn_state_encoder(input_size=rnn_input_size, hidden_size=hidden_size, rnn_type=model_config.STATE_ENCODER.rnn_type, num_layers=1)

        self.register_buffer("_scale", torch.tensor(1.0 / ((hidden_size // 2) ** 0.5)))
        self.state_q = nn.Linear(hidden_size, hidden_size // 2)
        
        self.text_k = nn.Conv1d(self.instruction_encoder.output_size, hidden_size // 2, 1)
        self.text_q = nn.Linear(self.instruction_encoder.output_size, hidden_size // 2)
        
        self.rgb_kv = nn.Conv1d(self.rgb_encoder.output_shape[0], hidden_size // 2 + model_config.RGB_ENCODER.output_size, 1)
        self.depth_kv = nn.Conv1d(self.depth_encoder.output_shape[0], hidden_size // 2 + model_config.DEPTH_ENCODER.output_size, 1)
        
        self.second_state_compress = nn.Sequential(
            nn.Linear(hidden_size + model_config.RGB_ENCODER.output_size + model_config.DEPTH_ENCODER.output_size + self.instruction_encoder.output_size + 32, hidden_size),
            nn.ReLU(True)
        )
        self.second_state_encoder = build_rnn_state_encoder(input_size=hidden_size, hidden_size=hidden_size, rnn_type=model_config.STATE_ENCODER.rnn_type, num_layers=1)

        self.cand_encoder = nn.Sequential(
            nn.Linear(4, 128),  # 注意这里是 4 (dist, sin, cos, valid)
            nn.ReLU(),
            nn.Linear(128, hidden_size)
        )
        self.stop_embedding = nn.Parameter(torch.randn(1, hidden_size))

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.rgb_encoder.is_blind or self.depth_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers + self.second_state_encoder.num_recurrent_layers

    def _attn(self, q, k, v, mask=None):
        logits = torch.einsum("nc, nci -> ni", q, k)
        if mask is not None:
            logits = logits - mask.float() * 1e8
        attn = F.softmax(logits * self._scale, dim=1)
        return torch.einsum("ni, nci -> nc", attn, v)

    def forward(self, observations, rnn_states, prev_actions, masks):
        # 兼容性处理
        if prev_actions.dim() > 1 and prev_actions.size(-1) > 1:
            current_prev_actions = prev_actions[:, 0]
        else:
            current_prev_actions = prev_actions
        current_prev_actions = current_prev_actions.long()

        # Depth & RGB Fix
        if "depth" in observations and observations["depth"].dim() == 3:
            observations["depth"] = observations["depth"].unsqueeze(-1)
        if "rgb" in observations and observations["rgb"].size(-1) == 4:
            observations["rgb"] = observations["rgb"][..., :3]

        instruction_embedding = self.instruction_encoder(observations)
        if instruction_embedding.dim() == 2:
            instruction_embedding = instruction_embedding.unsqueeze(1)
        if instruction_embedding.shape[-1] == self.instruction_encoder.output_size:
            text_mask = (instruction_embedding == 0.0).all(dim=2)
            instruction_embedding_t = instruction_embedding.permute(0, 2, 1)
        else:
            text_mask = (instruction_embedding == 0.0).all(dim=1)
            instruction_embedding_t = instruction_embedding

        depth_embedding = self.depth_encoder(observations)
        rgb_embedding = self.rgb_encoder(observations)
        
        x_rgb = self.rgb_linear(torch.flatten(rgb_embedding, 2))
        x_depth = self.depth_linear(torch.flatten(depth_embedding, 2))
        
        x_action = self.prev_action_embedding(((current_prev_actions.float() + 1) * masks).long().view(-1))
        
        state_in = torch.cat([x_rgb, x_depth, x_action], dim=1)
        rnn_states_out = rnn_states.detach().clone()
        (state, rnn_states_out[:, 0 : 1]) = self.state_encoder(state_in, rnn_states[:, 0 : 1], masks)

        text_state_q = self.state_q(state)
        text_state_k = self.text_k(instruction_embedding_t)
        text_embedding = self._attn(text_state_q, text_state_k, instruction_embedding_t, text_mask)
        
        rgb_k, rgb_v = torch.split(self.rgb_kv(torch.flatten(rgb_embedding, 2)), self._hidden_size // 2, dim=1)
        depth_k, depth_v = torch.split(self.depth_kv(torch.flatten(depth_embedding, 2)), self._hidden_size // 2, dim=1)
        text_q = self.text_q(text_embedding)
        
        rgb_attended = self._attn(text_q, rgb_k, rgb_v)
        depth_attended = self._attn(text_q, depth_k, depth_v)

        x = torch.cat([state, text_embedding, rgb_attended, depth_attended, x_action], dim=1)
        x = self.second_state_compress(x)
        (x, rnn_states_out[:, 1 : 2]) = self.second_state_encoder(x, rnn_states[:, 1 : 2], masks)
        
        cands = observations['candidate_waypoints'] 
        batch_size = cands.size(0)

        cand_embeds = self.cand_encoder(cands) 
        stop_embed = self.stop_embedding.expand(batch_size, 1, -1) 
        all_cands_embeds = torch.cat([cand_embeds, stop_embed], dim=1) 
        
        logits = torch.bmm(all_cands_embeds, x.unsqueeze(2)).squeeze(2)
        
        # ⚠️ REMOVED MASKING TO PREVENT LOSS INF ⚠️
        # 我们暂时移除了 Valid Mask 的强制过滤。
        # 这样即使 Expert 选了一个 "invalid" 点，Loss 也不会变成 Inf。
        # valid_mask = cands[:, :, 3] 
        # stop_mask = torch.ones(batch_size, 1, device=valid_mask.device)
        # full_mask = torch.cat([valid_mask, stop_mask], dim=1)
        # logits = logits.masked_fill(full_mask == 0, -float('inf'))

        return logits, rnn_states_out