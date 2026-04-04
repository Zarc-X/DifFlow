import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math

from .components.freia_flow import NormalizingFlowDensityEstimator

class FeatureCascadedAD(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        
        # 1. 前端：使用在 ImageNet 上预训练的 ResNet 作为强力特征提取器 (替代坍塌的纯随机 U-Net)
        # 冻结模型提取可靠的大规模多尺度的纹理特征用于异常定位
        self.cnn = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # 冻结所有特征提取器的参数，防止其发生特征模式坍塌 (Feature Collapse)
        for param in self.cnn.parameters():
            param.requires_grad = False
            
        self.cnn.eval() # 强行锁定 eval 模式 (锁定 BatchNorm 的均值和方差)
        
        # 提取 ResNet18 的 layer2 特征: 输出通道数为 128，空间尺寸降采样 8 倍 (例如输入256 -> 特征32x32)
        feature_dim = 128
        
        # 2. 后端：Normalizing Flow 密度估计 (针对高维度特征进行分布建模)
        self.density_estimator = NormalizingFlowDensityEstimator(in_channels=feature_dim, n_blocks=8)
        
    def extract_features(self, x):
        """
        手动前向传播截取 ResNet 内部特征图
        """
        self.cnn.eval() # 再次确保是 eval
        with torch.no_grad():
            x = self.cnn.conv1(x)
            x = self.cnn.bn1(x)
            x = self.cnn.relu(x)
            x = self.cnn.maxpool(x)
            x = self.cnn.layer1(x)  # [B, 64, 64, 64]
            x = self.cnn.layer2(x)  # [B, 128, 32, 32]
        return x

    def forward_train(self, x):
        """
        前向训练过程: 只训练 Normalize Flow
        """
        # 提取固定且可靠的预训练物理特征
        features = self.extract_features(x)
        
        # 传入流模型进行自然密度分布映射
        z, log_det = self.density_estimator(features)
        
        # 高斯先验对数似然
        prior_logprob = -0.5 * torch.sum(z**2, dim=[1,2,3]) - 0.5 * math.log(2*math.pi) * z[0].numel()
        
        # 【修复】统一除以全部特征单元数，保持损失在一个合理的刻度内平衡
        log_likelihood = (prior_logprob + log_det.view(-1)) / z[0].numel()
        
        # 最大化对数似然 = 最小化负对数似然
        loss_flow = -torch.mean(log_likelihood)
        
        # 这里实际上丢弃了 diffusion 部分（纯粹的经典 Flow- AD方案）。返回空占位符满足代码格式。
        return torch.tensor(0.0, device=x.device, requires_grad=True), loss_flow

    def forward_test(self, x):
        """
        前向推理过程（测试阶段定位异常）
        """
        features = self.extract_features(x)
        
        # 推理计算时不需要对流模型求导（我们直接基于特征算概率）
        with torch.no_grad():
            z, log_det = self.density_estimator(features)
        
        # 针对每个空间位置 (Pixel in Feature Map) 计算距离。
        # 这里用特征维度上的 z 平方和作为像素级别的距离度量（负均对数似然近示图）。
        # z 形状为 [B, C, H', W'], 对通道求平方和得到异常激活热力图
        local_anomaly_map = torch.sum(z**2, dim=1, keepdim=True)  # [B, 1, 32, 32]
        
        # 上采样回原图尺寸 [B, 1, 256, 256] 供后续做 Pixel AUROC 评定
        anomaly_score_map = F.interpolate(local_anomaly_map, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        return anomaly_score_map, x
