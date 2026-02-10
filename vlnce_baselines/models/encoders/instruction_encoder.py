import gzip
import json

import torch
import torch.nn as nn
from habitat import Config
from habitat.core.simulator import Observations
from torch import Tensor
from transformers import BertModel

class InstructionEncoder(nn.Module):
    def __init__(self, config: Config) -> None:
        """An encoder that uses RNN to encode an instruction. Returns
        the final hidden state after processing the instruction sequence.

        Args:
            config: must have
                embedding_size: The dimension of each embedding vector
                hidden_size: The hidden (output) size
                rnn_type: The RNN cell type.  Must be GRU or LSTM
                final_state_only: If True, return just the final state
        """
        super().__init__()

        self.config = config

        rnn = nn.GRU if self.config.rnn_type == "GRU" else nn.LSTM
        self.encoder_rnn = rnn(
            input_size=config.embedding_size,
            hidden_size=config.hidden_size,
            bidirectional=config.bidirectional,
        )

        if config.sensor_uuid == "instruction":
            if self.config.use_pretrained_embeddings:
                self.embedding_layer = nn.Embedding.from_pretrained(
                    embeddings=self._load_embeddings(),
                    freeze=not self.config.fine_tune_embeddings,
                )
            else:  # each embedding initialized to sampled Gaussian
                self.embedding_layer = nn.Embedding(
                    num_embeddings=config.vocab_size,
                    embedding_dim=config.embedding_size,
                    padding_idx=0,
                )

    @property
    def output_size(self):
        return self.config.hidden_size * (1 + int(self.config.bidirectional))

    def _load_embeddings(self) -> Tensor:
        """Loads word embeddings from a pretrained embeddings file.
        PAD: index 0. [0.0, ... 0.0]
        UNK: index 1. mean of all R2R word embeddings: [mean_0, ..., mean_n]
        why UNK is averaged: https://bit.ly/3u3hkYg
        Returns:
            embeddings tensor of size [num_words x embedding_dim]
        """
        with gzip.open(self.config.embedding_file, "rt") as f:
            embeddings = torch.tensor(json.load(f))
        return embeddings

    def forward(self, observations: Observations) -> Tensor:
        """
        Tensor sizes after computation:
            instruction: [batch_size x seq_length]
            lengths: [batch_size]
            hidden_state: [batch_size x hidden_size]
        """
        if self.config.sensor_uuid == "instruction":
            instruction = observations["instruction"].long()


            # ------------------ ✅ 最终修复代码 ------------------
            # 获取词表大小 (2504)
            vocab_size = self.embedding_layer.num_embeddings
            
            # 创建越界掩码
            # 凡是 >= 2504 的 ID 都是非法的
            out_of_bounds = instruction >= vocab_size
            
            # 如果发现越界，将其替换为 1 (UNK - Unknown Token)
            # 这样模型就会把它当成"一个不认识的词"来处理，而不会崩溃
            if out_of_bounds.any():
                # (可选) 如果你想保留日志，可以取消注释下面这行，只打印不报错
                print(f"⚠️ Warning: Clamping {out_of_bounds.sum()} tokens (>= {vocab_size}) to UNK.")
                instruction[out_of_bounds] = 1 
            # -----------------------------------------------------
            # # ------------------ 🔥 DEBUG 代码开始 🔥 ------------------
            # # 获取 Embedding 层的词表大小
            # vocab_size = self.embedding_layer.num_embeddings
            
            # # 获取当前 Batch 中最大的 Token ID
            # max_token_id = instruction.max().item()
            
            # # 检查是否越界
            # if max_token_id >= vocab_size:
            #     print(f"\n{'='*40}")
            #     print(f"🔥【严重错误】CUDA Device-side Assert Triggered 预警")
            #     print(f"🔥 检测到越界 Token ID: {max_token_id}")
            #     print(f"🔥 当前 Embedding 词表大小: {vocab_size}")
            #     print(f"🔥 越界位置 (Batch Index, Seq Index): {torch.nonzero(instruction >= vocab_size, as_tuple=False)}")
            #     print(f"{'='*40}\n")
                
            #     # 强制报错，阻止代码继续运行导致 CUDA 崩溃看不到日志
            #     raise ValueError(f"Found token {max_token_id} >= vocab size {vocab_size}")
            # # ------------------ 🔥 DEBUG 代码结束 🔥 ------------------

            lengths = (instruction != 0.0).long().sum(dim=1)
            instruction = self.embedding_layer(instruction)
        else:
            instruction = observations["rxr_instruction"]

        lengths = (instruction != 0.0).long().sum(dim=2)
        lengths = (lengths != 0.0).long().sum(dim=1).cpu()

        packed_seq = nn.utils.rnn.pack_padded_sequence(
            instruction, lengths, batch_first=True, enforce_sorted=False
        )

        output, final_state = self.encoder_rnn(packed_seq)

        if self.config.rnn_type == "LSTM":
            final_state = final_state[0]

        if self.config.final_state_only:
            return final_state.squeeze(0)
        else:
            return nn.utils.rnn.pad_packed_sequence(output, batch_first=True)[
                0
            ].permute(0, 2, 1)


class BertInstructionEncoder(nn.Module):
    def __init__(self, config: Config) -> None:
        """
        BERT Encoder with correct shape for CMAPolicy.
        """
        super().__init__()
        self.config = config

        print(f"Loading BERT Model (fine-tune={getattr(config, 'fine_tune_bert', True)})...")
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        # 🔥🔥🔥 关键在这里！Encoder 自己负责读取 fine_tune_bert 🔥🔥🔥
        # getattr(config, "key", Default) 的意思是：
        # 去 config 里找 "fine_tune_bert"，如果找不到，默认认为是 True
        self.fine_tune = getattr(config, "fine_tune_bert", True)

        # 执行冻结逻辑
        if not self.fine_tune:
            print("🥶 BERT Parameters Frozen (Like CLIP visual encoder)")
            for param in self.bert.parameters():
                param.requires_grad = False
        else:
            print("🔥 BERT Parameters Unfrozen (Fine-tuning enabled)")
        self.bert_dim = self.bert.config.hidden_size
        self.hidden_size = config.hidden_size
        
        self.projection = nn.Sequential(
            nn.Linear(self.bert_dim, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.attn_fc = nn.Sequential(
            nn.Linear(self.bert_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    @property
    def output_size(self):
        return self.hidden_size

    def forward(self, observations: Observations) -> Tensor:
        """
        Input: observations["instruction"] must be BERT Token IDs
        """
        input_ids = observations["instruction"].long()

        # # 🔥🔥🔥 DEBUG START 🔥🔥🔥
        # # 1. 打印形状：检查是否是你设置的固定长度（例如 120 或 128）
        # #    如果是变长（每个batch不一样），说明还在用旧 Sensor！
        # print(f"🧐 [Debug] Input Shape: {input_ids.shape}")

        # # 2. 打印最大 Token ID：
        # #    旧 R2R 词表最大 ID 只有 ~2500。
        # #    BERT 词表最大 ID 是 30522。
        # #    如果你看到 > 2500 的数字，说明肯定是 BERT Tokenizer 生效了！
        # max_id = input_ids.max().item()
        # print(f"🧐 [Debug] Max Token ID: {max_id}")
        
        # # 3. 打印前几个 Token：看看是不是 BERT 的特征（比如 101 开头）
        # #    101 是 BERT 的 [CLS] 标记。旧 R2R 数据集通常不会以 101 开头。
        # print(f"🧐 [Debug] First 5 tokens: {input_ids[0, :5].tolist()}")
        # # 🔥🔥🔥 DEBUG END 🔥🔥🔥

        attention_mask = (input_ids != 0).long()

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state # [Batch, Seq, 768]

        # 1. 投影到目标维度 [Batch, Seq, 256]
        out = self.projection(sequence_output)

        # 🔥🔥🔥 修正点 1: 手动将 Padding 区域抹零
        # 这一步至关重要！因为 CMAPolicy 里面通过 (emb == 0).all() 来判断哪些是 padding
        # 如果不抹零，Mask 就会失效，模型会把 Padding 也当成有效单词处理
        out = out * attention_mask.unsqueeze(-1).float()

        if self.config.final_state_only:
            # Attention Pooling 逻辑...
            attn_scores = self.attn_fc(sequence_output)
            mask = attention_mask.unsqueeze(-1).float()
            attn_scores = attn_scores + (1.0 - mask) * -1e9
            attn_weights = torch.softmax(attn_scores, dim=1)
            final_embed = torch.sum(sequence_output * attn_weights, dim=1)
            return self.projection(final_embed)
        else:
            # 🔥🔥🔥 修正点 2: 维度置换 (Permute)
            # CMAPolicy 里的 Conv1d 期望输入是 [Batch, Channel, Length]
            # 而 BERT 输出是 [Batch, Length, Channel]
            # 所以我们需要把维度 1 和 2 换一下
            return out.permute(0, 2, 1)