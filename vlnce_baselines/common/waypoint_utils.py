import numpy as np

def quaternion_rotate_vector(quat, v):
    """
    纯 Numpy 实现四元数旋转向量，避免依赖 habitat_sim 或 quaternion 库
    quat: [x, y, z, w] 或 [w, x, y, z] (Habitat 通常是 [x, y, z, w] 但 numpy 需要确认)
    Habitat Sim 的 agent_state.rotation 通常是 quaternion 对象
    """
    # 提取四元数分量
    # Habitat 的 quaternion 对象通常有 x, y, z, w 属性
    try:
        qx, qy, qz, qw = quat.x, quat.y, quat.z, quat.w
    except AttributeError:
        # 如果是数组/列表，假设顺序是 [x, y, z, w] (Habitat 惯例)
        if len(quat) == 4:
            qx, qy, qz, qw = quat[0], quat[1], quat[2], quat[3]
        else:
            return v # 无法处理，返回原向量

    # 构造旋转矩阵
    # https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
    x2 = qx + qx; y2 = qy + qy; z2 = qz + qz
    xx = qx * x2; xy = qx * y2; xz = qx * z2
    yy = qy * y2; yz = qy * z2; zz = qz * z2
    wx = qw * x2; wy = qw * y2; wz = qw * z2

    res = np.empty_like(v)
    res[0] = (1.0 - (yy + zz)) * v[0] + (xy - wz) * v[1] + (xz + wy) * v[2]
    res[1] = (xy + wz) * v[0] + (1.0 - (xx + zz)) * v[1] + (yz - wx) * v[2]
    res[2] = (xz - wy) * v[0] + (yz + wx) * v[1] + (1.0 - (xx + yy)) * v[2]
    
    return res

def get_candidate_waypoints(sim, num_candidates=12, max_distance=2.5, min_distance=0.5):
    """
    生成候选点 - 纯 Numpy 鲁棒版
    """
    try:
        # 1. 获取 Agent 状态
        agent_state = sim.get_agent(0).get_state()
        agent_position = np.array(agent_state.position, dtype=np.float32)
        agent_rotation = agent_state.rotation

        candidates = []
        # 0度是正前方 (-z)，顺时针生成
        angles = np.linspace(0, 2 * np.pi, num_candidates, endpoint=False)
        
        for angle in angles:
            # 2. 计算局部向量 (Habitat: -z is forward, x is right)
            # local_dir = [sin(angle), 0, -cos(angle)]
            local_dir = np.array([np.sin(angle), 0, -np.cos(angle)], dtype=np.float32)
            
            # 3. 旋转向量 (使用纯 Numpy 函数)
            global_dir = quaternion_rotate_vector(agent_rotation, local_dir)
            
            # 4. 寻找落脚点
            target_pos = agent_position + global_dir * max_distance
            
            # 检查 NaN
            if np.isnan(target_pos).any():
                continue

            # 调用 C++ 接口 (唯一可能崩溃的点，加上保护)
            snapped_point = sim.pathfinder.snap_point(target_pos)
            
            if np.isnan(snapped_point).any():
                is_navigable = False
                dist = 0.0
            else:
                is_navigable = sim.pathfinder.is_navigable(snapped_point)
                dist = np.linalg.norm(snapped_point - agent_position)
            
            # 5. 构造返回数据
            if is_navigable and dist >= min_distance:
                candidates.append({
                    "position": snapped_point,
                    "distance": float(dist),
                    "angle": float(angle),
                    "features": np.array([dist, np.sin(angle), np.cos(angle)], dtype=np.float32)
                })
            else:
                candidates.append({
                    "position": None,
                    "distance": 0.0,
                    "angle": float(angle),
                    "features": np.zeros(3, dtype=np.float32)
                })
                
        return candidates

    except Exception as e:
        print(f"🔥 Error in get_candidate_waypoints: {e}")
        dummy_feat = np.zeros(3, dtype=np.float32)
        return [{"position": None, "distance": 0.0, "angle": 0.0, "features": dummy_feat} for _ in range(num_candidates)]