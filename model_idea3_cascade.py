import torch
import torch.nn as nn

"""
思路三：特征级联/隐空间的流映射 (Diffusion Features + Flow Densities)

【架构思想】：网络级联（最强落地方案）。
这正是针对 MVTec AD 数据集最容易刷屏的高效框架。
1. 将 Diffusion 视为一个比 ResNet 更加强大、对缺陷更敏感的 “高级特征提取器”。
2. 给原图加上非常微弱的测试噪声，然后通过 U-Net 生成去噪中间特征 (通常选取 Decoder 的多尺度层)。
3. Flow 模型不再接收原始图像，而是接收 Diffusion 吐出来的高维张量特征。
4. Flow 模型负责对这些强有语义的正常样本特征构建复杂的概率密度函数。
5. 测试时，带有异常的缺陷部分会导致 Diffusion 输出扭曲的内部特征向量，进而被 Flow 诊断为极低密度的异常点。
"""

class ExtractorUNet(nn.Module):
    def __init__(self, channels=3, feature_dim=128):
        super().__init__()
        # 简化的 Encoder 部分
        self.enc = nn.Conv2d(channels, 64, kernel_size=3, padding=1)
        # 简化的 Decoder 部分 (关键的特征输出层)
        self.dec = nn.Conv2d(64, feature_dim, kernel_size=3, padding=1)
        
    def forward(self, x_noisy, t):
        # U-Net 内部过程，返回高维隐含特征 (不是最终的图像)
        h = torch.relu(self.enc(x_noisy))
        diff_features = torch.relu(self.dec(h)) # [B, feature_dim, H, W]
        return diff_features

class DensityEstimationFlow(nn.Module):
    def __init__(self, feature_dim=128):
        super().__init__()
        # 专门针对高维隐空间特征的 Normalizing Flow (例如 CFLOW / FastFlow 中的类似结构)
        self.flow_blocks = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 1), # 模拟流体变换块
            nn.LeakyReLU(), # 实际开发中不能用不可逆的ReLU，此处仅作逻辑演示
            nn.Conv2d(feature_dim, feature_dim, 1)
        )
        
    def forward(self, features):
        z = self.flow_blocks(features)
        # 伪装计算雅可比行列式
        log_det = torch.zeros(features.shape[0], 1, features.shape[2], features.shape[3], device=features.device)
        return z, log_det

class FeatureCascadedAD(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. 前端：Diffusion U-Net 提特征
        self.feature_extractor = ExtractorUNet(channels=3, feature_dim=128)
        
        # 2. 后端：Normalizing Flow 密度估计 (针对特征维度)
        self.density_estimator = DensityEstimationFlow(feature_dim=128)
        
    def forward_train(self, x):
        """
        前向训练过程
        为了兼容 train.py，我们计算一下重建或者Flow损失并返回
        """
        t_small = torch.tensor([1]).to(x.device)
        x_noisy = x + torch.randn_like(x) * 0.05
        diff_features = self.feature_extractor(x_noisy, t_small)
        z, log_det = self.density_estimator(diff_features)
        
        # 简单模拟最大似然目标 (实际还要算特征MSE等)
        loss = -torch.mean(-0.5 * torch.sum(z**2, dim=1) + log_det.squeeze(1))
        return loss, torch.tensor(0.0, device=x.device)

    def forward_test(self, x):
        """
        前向推理过程（测试阶段定位异常）
        """
        # 第一步：给输入图片施加轻微扩散噪声 (使得网络能够激活去噪先验机制)
        # t 非常小，不需要大量加噪
        t_small = torch.tensor([1]).to(x.device) 
        x_noisy = x + torch.randn_like(x) * 0.05
        
        # 第二步：获取 Diffusion 视角下的异常鉴别特征
        diff_features = self.feature_extractor(x_noisy, t_small) # Shape: [B, 128, H, W]
        
        # 【重要环节】：这里将 Diffusion 架构的输出 “桥接” 作为 Flow 架构的输入
        
        # 第三步：在特征隐空间通过 Flow 计算概率密度
        z, log_det = self.density_estimator(diff_features)
        
        # 第四步：计算特征映射到高斯隐空间后的 logp(z) 与 对数边缘概率
        log_p_z_features = -0.5 * torch.sum(z**2, dim=1, keepdim=True) # [B, 1, H, W]
        log_likelihood = log_p_z_features + log_det
        
        # 输出作为局部异常图
        anomaly_score_map = -log_likelihood
        
        import torch.nn.functional as F
        # 由于在隐空间操作，anomaly_score_map此时可能和原图尺寸不一致，需要用到上采样
        anomaly_score_map = F.interpolate(anomaly_score_map, size=x.shape[2:], mode='bilinear')
        
        return anomaly_score_map, x
