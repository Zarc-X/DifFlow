import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from types import SimpleNamespace

from .components.ddpm_unet import StandardDDPMUNet
from .components.flow.flow_models import build_msflow_model

class DiffusionMSFlowAD(nn.Module):
    """
    Idea 5: 融合 DiffusionAD 和 MSFlow
    采用单步降噪 (Single-step Denoising) 的扩散模型来充当三级特征提取器 (分别对应 t=100, 200, 400)。
    之后将这三个处于不同时序抽象层次的“伪原图(去噪预测图)”重采样为多尺度特征，
    喂入基于 FrEIA 的非对称多尺度归一化流 (MSFlow)。
    """
    def __init__(self, channels=3, n_steps=1000):
        super().__init__()
        
        # 1. 核心 U-Net，提供从各个 t 时刻对噪声情况的单步重构预测
        self.unet = StandardDDPMUNet(in_channels=channels, out_channels=channels, model_channels=64)
        
        # 关键修正：U-Net 未经预训练不能冻结，且需要端到端跟随流模型一起训练
        # 所以我们移除冻结代码，让 U-Net 的参数可以被优化
        # for param in self.unet.parameters():
        #     param.requires_grad = False
        # self.unet.train()

            
        self.n_steps = n_steps
        
        # 保护边界，假设配的 n_steps 如果不足 400，依然需要支持到至少最大 step
        if self.n_steps <= 400:
            self.n_steps = 401
            
        self.timesteps = [100, 200, 400]
        
        # DDPM 参数
        beta_start = 0.0001
        beta_end = 0.02
        betas = torch.linspace(beta_start, beta_end, self.n_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        
        # 2. 空间多尺度投影阵
        # 扩散网络预测的 x0 是原图尺寸 (256x256)，MSflow 需要多尺度 (例如 64, 32, 16)
        self.proj_stage1 = nn.Sequential(nn.Conv2d(channels, 64, kernel_size=4, stride=4), nn.BatchNorm2d(64), nn.ReLU())
        self.proj_stage2 = nn.Sequential(nn.Conv2d(channels, 128, kernel_size=8, stride=8), nn.BatchNorm2d(128), nn.ReLU())
        self.proj_stage3 = nn.Sequential(nn.Conv2d(channels, 256, kernel_size=16, stride=16), nn.BatchNorm2d(256), nn.ReLU())
        
        # 3. 建立 MSFlow
        c = SimpleNamespace(
            c_conds=[0, 0, 0],
            parallel_blocks=[4, 4, 4],  # 每尺度的 parallel 流个数
            clamp_alpha=1.9
        )
        c_feats = [64, 128, 256]
        self.c_feats = c_feats
        
        flows, self.fusion_flow = build_msflow_model(c, c_feats)
        self.parallel_flows = nn.ModuleList(flows)
        
    def _extract_multiscale_features(self, x_0, return_loss=False):
        """核心机制：基于不同时间的 Single-step Denoising 获取多阶段特征"""
        batch_size = x_0.shape[0]
        device = x_0.device
        
        multi_scale_features = []
        projs = [self.proj_stage1, self.proj_stage2, self.proj_stage3]
        
        diff_loss = 0.0
        
        for idx, t_val in enumerate(self.timesteps):
            t = torch.tensor([t_val] * batch_size, device=device, dtype=torch.long)
            
            # 前向加噪
            a_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
            noise = torch.randn_like(x_0)
            x_t = torch.sqrt(a_cumprod_t) * x_0 + torch.sqrt(1.0 - a_cumprod_t) * noise
            
            # 单步去噪预测，开启梯度用于反向传播
            pred_noise = self.unet(x_t, t.float())
            
            if return_loss:
                # 扩散标准 MSE 损失 (匹配预测噪声与真实噪声)
                diff_loss += F.mse_loss(pred_noise, noise)
            
            # 求解预测的 x_0 (Pred x_0)
            pred_x0 = (x_t - torch.sqrt(1.0 - a_cumprod_t) * pred_noise) / torch.sqrt(a_cumprod_t)
            
            # 投影到指定尺度
            # 【核心修复】：必须 detach，防止 Flow 的极大负对数似然梯度回传破坏 U-Net 的图像重建能力！
            pred_x0_detached = pred_x0.detach()
            
            # 投影到指定尺度
            feature = projs[idx](pred_x0_detached)
            multi_scale_features.append(feature)
            
        if return_loss:
            return multi_scale_features, diff_loss / len(self.timesteps)
        return multi_scale_features

    def forward_train(self, x):
        features, diff_loss = self._extract_multiscale_features(x, return_loss=True)
        
        # 走 Parallel Flow
        zs = []
        jac_parallels = []
        for i, p_flow in enumerate(self.parallel_flows):
            z_i, jac_i = p_flow(features[i])
            if isinstance(z_i, tuple) or isinstance(z_i, list):
                z_i = z_i[0]
            zs.append(z_i)
            jac_parallels.append(jac_i.view(-1))
            
        # 走 Fusion Flow
        z_fusion, jac_fusion = self.fusion_flow(zs)
        
        # 结合似然
        total_log_jac = sum(jac_parallels) + jac_fusion.view(-1)
        
        # 先验 P(z)
        prior_logprob = 0
        total_elements = 0
        for z_i in z_fusion:
            prior_logprob += -0.5 * torch.sum(z_i**2, dim=[1,2,3])
            total_elements += z_i[0].numel()
            
        # 与之前的保持一致
        prior_logprob = prior_logprob - 0.5 * math.log(2*math.pi) * total_elements
        
        log_likelihood = (prior_logprob + total_log_jac) / total_elements
        loss = -torch.mean(log_likelihood)
        
        return diff_loss, loss

    def forward_test(self, x):
        with torch.no_grad():
            features = self._extract_multiscale_features(x, return_loss=False)
            
            zs = []
            for i, p_flow in enumerate(self.parallel_flows):
                z_i, _ = p_flow(features[i])
                if isinstance(z_i, tuple) or isinstance(z_i, list):
                    z_i = z_i[0]
                zs.append(z_i)
            
            z_fusion, _ = self.fusion_flow(zs)
            
        # 基于 MSFlow 常规逻辑：多层级的隐变量求欧氏距离作为异常得分，然后双线性插值合并
        anomaly_map = None
        for z_i in z_fusion:
            # 必须从求局部平方和(sum)改为局部平方均值(mean)，来抵消 256 大通道特征掩盖细粒度 64 通道异常的数值悬殊！
            local_map = torch.mean(z_i**2, dim=1, keepdim=True)
            # 恢复到原图大小
            rescaled_map = F.interpolate(local_map, size=x.shape[2:], mode='bilinear', align_corners=False)
            if anomaly_map is None:
                anomaly_map = rescaled_map
            else:
                anomaly_map += rescaled_map
                
        # 由于我们只关注异常发现，忽略额外项
        return anomaly_map, x
