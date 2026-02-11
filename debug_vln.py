import habitat
from habitat.config import Config
# 🔥🔥🔥 必须添加这两行！🔥🔥🔥
# 只有先执行注册，Habitat 才能识别配置文件里的 CANDIDATE_MOVE_0
from vlnce_baselines.common.candidate_actions import register_candidate_actions
register_candidate_actions()
from habitat.core.env import Env
from vlnce_baselines.common.env_utils import construct_envs
from vlnce_baselines.config.default import get_config
import os

# 1. 强制设置环境变量，确保能看到 GLOG
os.environ["HABITAT_SIM_LOG"] = "verbose"
os.environ["MAGNUM_LOG"] = "verbose"
os.environ["GLOG_minloglevel"] = "0"

def debug_main():
    # 2. 加载你的配置文件
    config_path = "vlnce_baselines/config/r2r_baselines/vlnce_candidate.yaml"
    
    print(f"Loading config from: {config_path}")
    config = get_config(config_path)
    
    # 3. 强制修改配置，确保单线程、单环境、无干扰
    config.defrost()
    config.SYSTEM.NUM_PROCESSES = 1
    config.SIMULATOR_GPU_IDS = [0]
    config.IL.batch_size = 1
    # 确保没有重复 Sensor
    if "INSTRUCTION_SENSOR" in config.TASK_CONFIG.TASK.SENSORS:
        # 去重逻辑
        config.TASK_CONFIG.TASK.SENSORS = list(set(config.TASK_CONFIG.TASK.SENSORS))
    config.freeze()

    print("----------------------------------------------------------------")
    print("🛠️  正在尝试直接初始化单个环境 (Bypassing VectorEnv)...")
    print("----------------------------------------------------------------")

    try:
        # 4. 直接初始化 Habitat Env (不是 VectorEnv)
        # 这会直接调用底层 C++，如果有错，会当场报错
        env = Env(config=config.TASK_CONFIG)
        
        print("✅ 环境初始化成功！")
        print("尝试 reset()...")
        
        # 5. 尝试 Reset (最容易崩的地方)
        obs = env.reset()
        print("✅ Reset 成功！")
        
        # 打印一下拿到的观测数据 Keys
        print("观测数据 Keys:", obs.keys())
        
        if "depth" in obs:
            print("Depth Shape:", obs["depth"].shape)
        if "rgb" in obs:
            print("RGB Shape:", obs["rgb"].shape)

        env.close()
        print("🎉 恭喜！单环境运行正常。问题确实出在多进程本身。")

    except Exception as e:
        print("\n\n❌ 捕捉到 Python 异常:")
        print(e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_main()