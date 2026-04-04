import os
import sys
import torch
import torch.nn as nn

from .sdas.unets import UNetModel, EncoderUNetModel

class StandardDDPMUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, model_channels=64):
        super().__init__()
        # 使用 realnet 中的成熟多头注意力 UNet
        self.unet = UNetModel(
            image_size=256,
            in_channels=in_channels,
            model_channels=model_channels,
            out_channels=out_channels,
            num_res_blocks=2,
            attention_resolutions=(8, 16), # 修改了这里的注意力分辨率，原本在2,4(即128x128, 64x64)分辨率触发注意力极大占用显存
            dropout=0.1,
            channel_mult=(1, 2, 4, 8),
            num_classes=None,
            use_fp16=False,
            num_heads=1,  # 只有在小分辨率特征图上用或降低头数
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=True,
            resblock_updown=True,
        )

    def forward(self, x, t):
        # 兼容当前传入的一维标量时间步
        return self.unet(x, t)

class AdvancedExtractorUNet(nn.Module):
    def __init__(self, in_channels=3, feature_dim=128):
        super().__init__()
        # 用于 Idea 3 提取特征的 Encoder
        self.encoder = EncoderUNetModel(
            image_size=256,
            in_channels=in_channels,
            model_channels=64,
            out_channels=feature_dim,
            num_res_blocks=2,
            attention_resolutions=(8, 16), # 修小注意力分辨率以防 OOM
            dropout=0.1,
            num_heads=1, # 降低注意力头数
            channel_mult=(1, 2, 4),
            pool="none" # 必须返回 4D 张量 (B, C, H, W) 才能输入进后面的 Normalizing Flow
        )
        self.feature_dim = feature_dim

    def forward(self, x, t):
        return self.encoder(x, t)
