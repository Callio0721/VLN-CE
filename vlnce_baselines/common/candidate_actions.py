import habitat_sim
from habitat.core.registry import registry
from habitat.core.embodied_task import SimulatorTaskAction
from vlnce_baselines.common.waypoint_utils import get_candidate_waypoints

class CandidateMoveActionBase(SimulatorTaskAction):
    def step(self, *args, **kwargs):
        # 1. 解析动作索引
        try:
            cand_idx = int(self.name.split("_")[-1])
        except:
            cand_idx = 0

        # 2. 获取候选点
        cands = get_candidate_waypoints(self._sim, num_candidates=12)
        
        # 3. 越界或无效保护 -> 原地不动
        if cand_idx >= len(cands) or cands[cand_idx]["position"] is None:
            # 🔥 修复点 1: 直接获取观测，不调用 step("STOP") 防止崩溃
            return self._sim.get_sensor_observations()

        cand = cands[cand_idx]
        target_pos = cand["position"]
        
        # 4. 执行瞬间移动 (Teleport)
        agent = self._sim.get_agent(0)
        new_state = agent.get_state()
        new_state.position = target_pos
        agent.set_state(new_state)
        
        # 5. 刷新传感器
        # 🔥 修复点 2: 关键修改！不要调用 self._sim.step("STOP")
        # 直接读取传感器数据，这在 Habitat 中是安全的，不会触发物理引擎计算
        return self._sim.get_sensor_observations()

def register_candidate_actions():
    print("🚀 Registering Custom Candidate Actions (0-11)...")
    for i in range(12):
        action_name = f"CANDIDATE_MOVE_{i}"
        
        try:
            cls = type(
                f"CandidateMoveAction{i}", 
                (CandidateMoveActionBase,), 
                {"name": action_name}
            )
            registry.register_task_action(cls, name=action_name)
        except (AssertionError, ValueError):
            pass
    print("✅ Candidate Actions Registration Complete.")