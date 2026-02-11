import torch
from torch.utils.data import Dataset, DataLoader
import lmdb
import msgpack_numpy
import numpy as np
import tqdm
import os

# ---------------- 配置区域 ----------------
lmdb_path = "/home/ShiKaituo/ZhangBodong/VLN-CE/data/trajectories_dirs/cma_clip_pm_da_aug_tune/trajectories.lmdb"
vocab_size_limit = 2502
# ----------------------------------------

class VocabCheckDataset(Dataset):
    def __init__(self, lmdb_path):
        self.lmdb_path = lmdb_path
        print(f"📖 [阶段1/2] 正在初始化 Dataset...")
        
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False)
        with self.env.begin() as txn:
            self.length = txn.stat()['entries']
            # 🔥 优化：给 Key 的加载过程也加上进度条，这样你就知道没卡死
            with txn.cursor() as curs:
                self.keys = []
                # 这里的 tqdm 会显示读取 Key 的进度
                for key, _ in tqdm.tqdm(curs, total=self.length, desc="Loading Keys", unit="it"):
                    self.keys.append(key)
        self.env.close()
        self.env = None 

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self.env is None:
            self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False)
        
        key = self.keys[index]
        with self.env.begin() as txn:
            value = txn.get(key)
        
        try:
            item = msgpack_numpy.unpackb(value, raw=False)
            if isinstance(item, (list, tuple)) and len(item) > 0:
                obs = item[0]
                if 'instruction' in obs:
                    instr = obs['instruction']
                    return np.max(instr).item()
        except Exception:
            pass
        return 0 

def check_vocab_limit_torch():
    if not os.path.exists(lmdb_path):
        print(f"❌ 错误：路径不存在 -> {lmdb_path}")
        return

    # 1. 初始化 Dataset (这里会显示第一个进度条)
    dataset = VocabCheckDataset(lmdb_path)

    print(f"\n🚀 [阶段2/2] 开始多进程扫描 (Workers: {os.cpu_count()})...")
    
    # 2. 创建 DataLoader
    dataloader = DataLoader(
        dataset, 
        batch_size=2048, # 调大 Batch Size 让进度条跑得更顺滑
        shuffle=False, 
        num_workers=os.cpu_count(),
        collate_fn=lambda x: max(x) 
    )

    global_max = 0
    
    # 3. 扫描过程 (这里会显示第二个进度条)
    # unit="batch" 让你知道处理了多少个批次
    for batch_max in tqdm.tqdm(dataloader, desc="Scanning Tokens", unit="batch"):
        if batch_max > global_max:
            global_max = batch_max
            
    print("\n" + "="*40)
    print(f"🔢 扫描结果 - 最大 Token ID: {global_max}")
    print(f"🛡️ 模型 Embedding 大小: {vocab_size_limit}")
    
    if global_max < vocab_size_limit:
        print("\n🎉 验证成功！所有 Token 都在安全范围内。")
    else:
        print(f"\n❌ 验证失败！最大 ID {global_max} 超过了限制。")
        print("💡 必须保留 InstructionEncoder 里的截断代码。")
    print("="*40)

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)
    check_vocab_limit_torch()