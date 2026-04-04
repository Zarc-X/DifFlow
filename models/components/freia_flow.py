import os
import sys
import torch
import torch.nn as nn

from .flow.flow_models import single_parallel_flows, subnet_conv_ln

class NormalizingFlowDensityEstimator(nn.Module):
    """
    基于 FrEIA 和 msflow 中的 RealNVP/AllInOneBlock 构建可逆流体模型
    """
    def __init__(self, in_channels=3, n_blocks=4, clamp_alpha=1.9):
        super().__init__()
        self.in_channels = in_channels
        self.n_blocks = n_blocks
        # cond_shape 在 msflow 里设为 0 如果没有 condition。为了简单，直接无 condition
        c_cond = 0
        
        # 建立并行序列流
        self.flow = single_parallel_flows(
            c_feat=in_channels, 
            c_cond=c_cond, 
            n_block=n_blocks, 
            clamp_alpha=clamp_alpha, 
            subnet=subnet_conv_ln
        )
        
    def forward(self, x):
        # FrEIA 流前向传播，得到潜在变量 z 和 雅可比对数行列式 jac
        # output is: [z], jac
        z, jac = self.flow(x)
        if isinstance(z, list) or isinstance(z, tuple):
            z = z[0]
        
        # 求似然概率 = prior(z) + jac
        # 其中 prior 假设为独立标准正态分布: log N(z; 0, I) = -0.5 * z^2 - 0.5 * log(2*pi)
        log_prior_z = -0.5 * torch.sum(z**2, dim=1, keepdim=True)
        
        # 为了兼容，返回 log_likelihood 或者 (z, jac)  tùy调用者
        return z, jac

