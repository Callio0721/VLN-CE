import os
import habitat_sim

print("🔍 --- 环境诊断开始 ---")
print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not Set')[:50]}...") # 只打前50个字符防刷屏
print(f"JSON Config: {os.environ.get('__EGL_VENDOR_LIBRARY_FILENAMES', '⚠️ 致命错误: 未设置')}")

try:
    # 1. 配置后端
    backend_cfg = habitat_sim.SimulatorConfiguration()
    
    # 2. 配置 Agent (修复之前的报错)
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    
    # 3. 组合配置
    # 注意：这里必须把 agent_cfg 放入列表中传给 Configuration
    cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
    
    # 4. 初始化仿真器 (最关键的一步，驱动如果坏了会在这里崩)
    print("⏳ 正在尝试初始化 Simulator...")
    sim = habitat_sim.Simulator(cfg)
    
    print("🎉 EGL 初始化成功！显卡驱动工作正常！")
    
    # 简单测试一下渲染
    print(f"当前场景 ID: {sim.curr_scene_name}")
    sim.close()
    
except Exception as e:
    print(f"❌ Python 层面捕获到错误: {e}")

print("🔍 --- 诊断结束 ---")