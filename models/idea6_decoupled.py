import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from types import SimpleNamespace

from .components.flow.flow_models import build_msflow_model

class DecoupledDiffFlowAD(nn.Module):
    def __init__(self, channels=3, n_steps=400):
        super().__init__()
        
        # 1. “换心手术”：使用预训练的 ResNet18 作为极其强大的静态语义特征提取器
        # 完全抛弃从零训练且容易出现域偏移灾难的 Diffusion U-Net
        resnet = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,     # 1/4
            resnet.layer1,      # 1/4 size (64x64 for 256x256 input), 64 channels
            resnet.layer2,      # 1/8 size (32x32), 128 channels
            resnet.layer3       # 1/16 size (16x16), 256 channels
        )
        
        # 严格冻结特征提取器的所有参数 (Backbone 绝不参与训练)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor.eval()

        # 注册勾子提取多层级特征
        self.feature_maps = []
        def get_activation():
            def hook(model, input, output):
                self.feature_maps.append(output)
            return hook
            
        # [4] 是 layer1, [5] 是 layer2, [6] 是 layer3
        self.feature_extractor[4].register_forward_hook(get_activation()) 
        self.feature_extractor[5].register_forward_hook(get_activation()) 
        self.feature_extractor[6].register_forward_hook(get_activation()) 
        
        # 2. 自适应核函数的 MSFlow
        self.c_feats = [64, 128, 256]
        c = SimpleNamespace(
            c_conds=[0, 0, 0],
            # 浅层特征维度低但细节繁杂：提供 8 个流块重兵压阵
            # 中层特征居中：提供 4 个流块
            # 深层特征极度抽象且接近高斯形态：仅需 2 个流块即可收敛，防止过拟合
            parallel_blocks=[8, 4, 2],  
            clamp_alpha=1.9
        )
        
        parallel_flows, _ = build_msflow_model(c, self.c_feats)
        self.parallel_flows = nn.ModuleList(parallel_flows)
        # 完全抛弃破坏空间分辨率的 Fusion Flow 层
        # self.fusion_flow = fusion_flow
        
    def _extract_features(self, x):
        self.feature_maps = []
        with torch.no_grad():
            self.feature_extractor(x)
        return self.feature_maps

    def forward_train(self, x_orig, epoch=None):
        # 保证特征骨干绝对在 eval 状态，不更新 Batch Norm 统计量
        self.feature_extractor.eval()
        
        # 直接输出高维语义特征
        feats = self._extract_features(x_orig)
        
        loss_flow = 0.0
        zs = []
        
        # 仅针对 Flow 进行负对数似然优化
        for i, (p_flow, feat) in enumerate(zip(self.parallel_flows, feats)):
            z_i, jac_i = p_flow(feat)
            if isinstance(z_i, list) or isinstance(z_i, tuple): z_i = z_i[0]
            
            # 【致命修复】：必须在全体空间通道维度上求和对数似然，然后再除以像素总数均摊
            # 否则 jac_i 的量级(H*W*C)会直接吞噬压垮 z_i平方和(均值仅为C)
            prior_logprob = -0.5 * torch.sum(z_i**2, dim=[1, 2, 3])
            log_likelihood = prior_logprob + jac_i
            loss_flow += -torch.mean(log_likelihood) / z_i[0].numel()
            
            zs.append(z_i)
            
        # 返回 dummy loss_diff(0.0) 以兼容外部 train.py 接口逻辑
        return torch.tensor(0.0, device=x_orig.device, requires_grad=True), loss_flow

    def forward_test(self, x_orig):
        self.feature_extractor.eval()
        feats = self._extract_features(x_orig)
        
        zs = []
        for i, p_flow in enumerate(self.parallel_flows):
            z_i, _ = p_flow(feats[i])
            if isinstance(z_i, tuple) or isinstance(z_i, list): z_i = z_i[0]
            zs.append(z_i)
            
        anomaly_map = None
        for z_i in zs:
            # 取各尺度潜空间平方均值计算局部密度异常概率
            local_map = torch.mean(z_i**2, dim=1, keepdim=True)
            # 通过双线性插值直接打回到原图尺寸 (256x256)
            rescaled_map = F.interpolate(local_map, size=x_orig.shape[2:], mode='bilinear', align_corners=False)
            
            if anomaly_map is None:
                anomaly_map = rescaled_map
            else:
                anomaly_map += rescaled_map
                
        return anomaly_map, x_orig
