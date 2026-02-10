import torch

# 替换成你那个报错的 pth 文件路径
pth_path = "/home/ShiKaituo/ZhangBodong/VLN-CE/data/checkpoints/cma_clip_pm_aug/ckpt.29.pth" 

try:
    # 加载文件
    ckpt = torch.load(pth_path, map_location='cpu')
    print(f"✅ 成功加载文件: {pth_path}")
    
    # 检查最外层的数据类型
    if isinstance(ckpt, dict):
        print(f"📦 文件结构是字典 (Dict)，包含的 Keys 有: {list(ckpt.keys())}")
        
        # 检查是否包含关键的 'state_dict' 或 'model'
        if 'state_dict' in ckpt:
            print("   -> 包含 'state_dict'，这是一个标准的训练检查点。")
            # 打印权重的第一个 key 看看前缀
            first_key = list(ckpt['state_dict'].keys())[0]
            print(f"   -> 权重 Key 示例: {first_key}")
        elif 'epoch' not in ckpt:
            # 有可能是纯权重字典，但不仅仅是 state_dict
            print("   -> ⚠️ 警告：这是一个字典，但没有发现 'epoch' 或 'state_dict' 键。这可能就是单纯的权重字典。")
            
    else:
        # 如果不是字典，可能是什么奇怪的对象
        print(f"⚠️ 文件结构不是字典，而是: {type(ckpt)}")

except Exception as e:
    print(f"❌ 加载失败: {e}")