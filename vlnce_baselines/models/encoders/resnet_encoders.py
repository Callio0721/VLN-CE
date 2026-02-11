from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from gym import Space, spaces
from habitat.core.simulator import Observations
from habitat_baselines.rl.ddppo.policy import resnet
from habitat_baselines.rl.ddppo.policy.resnet_policy import ResNetEncoder
from torch import Tensor

from vlnce_baselines.common.utils import single_frame_box_shape

# --- 在 resnet_encoders.py 头部添加 import ---
from transformers import CLIPVisionModel
import torchvision.transforms as T


class VlnResnetDepthEncoder(nn.Module):
    def __init__(
        self,
        observation_space: Space,
        output_size: int = 128,
        checkpoint: str = "NONE",
        backbone: str = "resnet50",
        resnet_baseplanes: int = 32,
        normalize_visual_inputs: bool = False,
        trainable: bool = False,
        spatial_output: bool = False,
    ) -> None:
        super().__init__()

        self.visual_encoder = ResNetEncoder(
            spaces.Dict(
                {
                    "depth": single_frame_box_shape(
                        observation_space.spaces["depth"]
                    )
                }
            ),
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes // 2,
            make_backbone=getattr(resnet, backbone),
            normalize_visual_inputs=normalize_visual_inputs,
        )

        for param in self.visual_encoder.parameters():
            param.requires_grad_(trainable)

        if checkpoint != "NONE":
            ddppo_weights = torch.load(checkpoint)

            weights_dict = {}
            for k, v in ddppo_weights["state_dict"].items():
                split_layer_name = k.split(".")[2:]
                if split_layer_name[0] != "visual_encoder":
                    continue

                layer_name = ".".join(split_layer_name[1:])
                weights_dict[layer_name] = v

            del ddppo_weights
            self.visual_encoder.load_state_dict(weights_dict, strict=True)

        self.spatial_output = spatial_output

        if not self.spatial_output:
            self.output_shape = (output_size,)
            self.visual_fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(
                    np.prod(self.visual_encoder.output_shape), output_size
                ),
                nn.ReLU(True),
            )
        else:
            self.spatial_embeddings = nn.Embedding(
                self.visual_encoder.output_shape[1]
                * self.visual_encoder.output_shape[2],
                64,
            )

            self.output_shape = list(self.visual_encoder.output_shape)
            self.output_shape[0] += self.spatial_embeddings.embedding_dim
            self.output_shape = tuple(self.output_shape)

    def forward(self, observations: Observations) -> Tensor:
        """
        Args:
            observations: [BATCH, HEIGHT, WIDTH, CHANNEL]
        Returns:
            [BATCH, OUTPUT_SIZE]
        """
        if "depth_features" in observations:
            x = observations["depth_features"]
        else:
            x = self.visual_encoder(observations)

        if self.spatial_output:
            b, c, h, w = x.size()

            spatial_features = (
                self.spatial_embeddings(
                    torch.arange(
                        0,
                        self.spatial_embeddings.num_embeddings,
                        device=x.device,
                        dtype=torch.long,
                    )
                )
                .view(1, -1, h, w)
                .expand(b, self.spatial_embeddings.embedding_dim, h, w)
            )

            return torch.cat([x, spatial_features], dim=1)
        else:
            return self.visual_fc(x)


class TorchVisionResNet(nn.Module):
    """TorchVision ResNet pre-trained on ImageNet. The standard average
    pooling can be replaced with spatial average pooling. The final fc layer
    is replaced with a new fc layer of a specified output size.
    """

    def __init__(
        self,
        output_size: int,
        resnet_version: str = "resnet50",
        normalize_visual_inputs: bool = False,
        trainable: bool = False,
        spatial_output: bool = False,
        single_spatial_filter: bool = True,
    ) -> None:
        super().__init__()
        self.normalize_visual_inputs = normalize_visual_inputs
        self.spatial_output = spatial_output
        resnet = getattr(models, resnet_version)(pretrained=True)
        modules = list(resnet.children())
        self.resnet_layer_size = modules[-1].in_features
        self.cnn = nn.Sequential(*modules[:-1])

        for param in self.cnn.parameters():
            param.requires_grad_(trainable)
        self.cnn.train(trainable)

        if not self.spatial_output:
            self.output_shape = (output_size,)
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.resnet_layer_size, output_size),
                nn.ReLU(),
            )
        else:

            class SpatialAvgPool(nn.Module):
                def forward(self, x):
                    x = F.adaptive_avg_pool2d(x, (4, 4))

                    return x

            if single_spatial_filter:
                self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])
            self.cnn.avgpool = SpatialAvgPool()
            self.spatial_embeddings = nn.Embedding(4 * 4, 64)
            self.output_shape = (
                self.resnet_layer_size + self.spatial_embeddings.embedding_dim,
                4,
                4,
            )

    def forward(self, observations: Observations) -> Tensor:
        def normalize(imgs: Tensor) -> Tensor:
            """Normalizes a batch of images by:
                1) scaling pixel values to be in the range 0-1
                2) subtracting the ImageNet mean
                3) dividing by the ImageNet variance
                TODO: could be nice to calculate mean and variance for Habitat MP3D scenes.
                    Method: compute for training split with oracle path follower.
            Args:
                imgs: must have pixel values ranging from 0-255. Assumes a size of [Bx3xHxW]
            https://github.com/pratogab/batch-transforms/blob/master/batch_transforms.py
            """
            imgs = imgs.contiguous() / 255.0
            if self.normalize_visual_inputs:
                mean_norm = torch.tensor([0.485, 0.456, 0.406]).to(
                    device=imgs.device
                )[None, :, None, None]
                std_norm = torch.tensor([0.229, 0.224, 0.225]).to(
                    device=imgs.device
                )[None, :, None, None]
                return imgs.sub(mean_norm).div(std_norm)
            else:
                return imgs

        if "rgb_features" in observations:
            resnet_output = observations["rgb_features"]
        else:
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT x WIDTH]
            rgb_observations = observations["rgb"].permute(0, 3, 1, 2)
            resnet_output = self.cnn(normalize(rgb_observations))

        if self.spatial_output:
            b, c, h, w = resnet_output.size()

            spatial_features = (
                self.spatial_embeddings(
                    torch.arange(
                        0,
                        self.spatial_embeddings.num_embeddings,
                        device=resnet_output.device,
                        dtype=torch.long,
                    )
                )
                .view(1, -1, h, w)
                .expand(b, self.spatial_embeddings.embedding_dim, h, w)
            )

            return torch.cat([resnet_output, spatial_features], dim=1)
        else:
            return self.fc(resnet_output)  # returns [BATCH x OUTPUT_DIM]


class TorchVisionResNet50(TorchVisionResNet):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, resnet_version="resnet50", **kwargs)


class TorchVisionResNet18(TorchVisionResNet):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, resnet_version="resnet18", **kwargs)


# --- 在文件末尾添加这个类 ---
class ClipVisualEncoder(nn.Module):
    def __init__(
        self, 
        output_size,   # <--- 改动点：必须放在第一个，匹配 CMAPolicy 的调用
        observation_space=None, # 改动点：变成可选参数，或者直接删掉，用 kwargs 接收
        model_name="/home/ShiKaituo/ZhangBodong/VLN-CE/clip-vit-base-patch32",
        trainable=False, 
        **kwargs       # <--- 这里会帮你“吞掉” normalize_visual_inputs 和 spatial_output 等多余参数，防止报错
    ):
        super().__init__()
        print(f"Loading CLIP: {model_name}, Output Size: {output_size}")
        
        self.backbone = CLIPVisionModel.from_pretrained(model_name)
        
        # -----------------------------------------------------------
        # 关键逻辑：参数冻结策略
        # -----------------------------------------------------------
        # 1. 首先，为了节省显存，我们总是冻结 CLIP 的主干 (Backbone)
        # 不管配置文件怎么写，CLIP 这种大模型在 VLN 里一般都跑不动微调
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # 2. 定义投影层：把 CLIP 的 768 变成 Config 要求的 256
        self.projection = nn.Linear(768, output_size)
        
        # 3. 处理 trainable 配置
        # 如果 config 说 trainable=True，那是为了训练 projection 层。
        # 如果 config 说 trainable=False，那 projection 层也就没法训练了 (这是不对的)
        # 所以，我们在这里做一个强制覆盖，或者依赖 config 必须设为 True。
        
        # 建议：Projection 层永远需要训练，否则它就是随机噪声
        # 即使 trainable=False，我们依然让 projection 可导，
        # 但这取决于 Optimizer 是否会扫描到它。
        # 为了安全起见，我们在 yaml 里必须把 trainable 设为 True。

        self.output_size = output_size
        self._output_shape = (output_size, 49) 

        # CLIP 的标准化
        # 注册 CLIP 的均值和方差，形状设为 [1, 3, 1, 1] 以便利用广播机制直接处理 Batch
        self.register_buffer(
            "clip_mean", 
            torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "clip_std", 
            torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
        )

    @property
    def is_blind(self):
        return False
        
    @property
    def output_shape(self):
        return self._output_shape


    def forward(self, observations):
            """
            observations["rgb"]: [Batch, H, W, 3] (uint8 0-255)
            """
            # 🔥🔥🔥 新增逻辑：优先检查是否有预计算的特征 🔥🔥🔥
            # 如果 DAgger 已经把 Backbone 的特征跑出来并存好了，直接用！
            if "rgb_features" in observations:
                features = observations["rgb_features"]
            else:
                # 只有没缓存的时候，才跑 CLIP Backbone
                rgb = observations["rgb"]
                
                # 1. 数据预处理
                # [B, H, W, 3] -> [B, 3, H, W] -> float -> 0-1
                rgb = rgb.permute(0, 3, 1, 2).float() / 255.0
                
                # 应用 CLIP 标准化
                rgb = (rgb - self.clip_mean) / self.clip_std

                # 2. CLIP 前向传播
                outputs = self.backbone(pixel_values=rgb)
                features = outputs.last_hidden_state

            # 3. 提取特征 (不管是缓存的还是新算的，格式都是 [Batch, 50, 768])
            # 去掉第一个 [CLS] token (索引0)，保留后面的 spatial patches (索引1-49)
            spatial_features = features[:, 1:, :]
            
            # 4. 降维投影 (这个 Projection 层是你要训练的！)
            spatial_features = self.projection(spatial_features)

            # 5. 调整形状
            return spatial_features.permute(0, 2, 1)
    # def forward(self, observations):
    #         """
    #         observations["rgb"]: [Batch, H, W, 3] (uint8 0-255)
    #         """
    #         rgb = observations["rgb"]
            
    #         # 1. 数据预处理
    #         # [B, H, W, 3] -> [B, 3, H, W] -> float -> 0-1
    #         rgb = rgb.permute(0, 3, 1, 2).float() / 255.0
            
    #         # 应用 CLIP 标准化
    #         # 2. 手动归一化 (直接用 Tensor 计算，支持 Batch)
    #         # 删掉: rgb = self.preprocess(rgb)
    #         rgb = (rgb - self.clip_mean) / self.clip_std

    #         # 2. CLIP 前向传播
    #         # 输出: [Batch, Seq_Len(50), Hidden(768)]
    #         outputs = self.backbone(pixel_values=rgb)
    #         features = outputs.last_hidden_state

    #         # 3. 提取特征
    #         # 去掉第一个 [CLS] token (索引0)，保留后面的 spatial patches (索引1-49)
    #         # features 变成 [Batch, 49, 768]
    #         spatial_features = features[:, 1:, :]
            
    #         # 4. 降维投影
    #         # [Batch, 49, 768] -> [Batch, 49, 512]
    #         spatial_features = self.projection(spatial_features)

    #         # 5. 调整形状以适配 CMANet
    #         # CMANet 期望的是 [Batch, Channel, Spatial/Seq]
    #         # 所以我们需要转置: [Batch, 512, 49]
    #         return spatial_features.permute(0, 2, 1)