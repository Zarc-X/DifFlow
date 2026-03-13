import torch
import torch.nn as nn
import torch.nn.functional as F

"""
思路一：每一步交替结合 / 去噪过程与概率分布引导 (Guided DiffFlow for AD)

【架构思想】：此方案严格遵循“在反向去噪的每一步中都结合 Flow 模型进行映射/降噪”。
1. 前向过程：和标准Diffusion一样，对原图逐渐加燥。
2. 反向过程：在每一步 t -> t-1 去噪时：
   a. 使用 Diffusion U-Net 预测基础的去噪步长（重建方向）。
   b. 将当前带燥特征/图像送入 Normalizing Flow 计算精确的似然概率 P_flow(x_t)。
   c. 计算出 Flow 对输入 x_t 的梯度：∇_x log P_flow(x_t)。
   d. 把这个梯度作为一个“引导场（Guidance）”，加进 Diffusion 的这一步去噪中。
3. 效果：在每一步，Flow 都在“硬拽”图片，使其强行落回正常工业零件的概率流形(Manifold)上；
         那些难以被拽回的结构（即 Flow 梯度给出的纠正量极大的区域），可以直接累加作为异常得分。
"""

class DiffUNet(nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        # 简化版U-Net，输入通道包含图像和时间步t
        self.conv = nn.Conv2d(channels + 1, channels, 3, padding=1)
        
    def forward(self, x, t):
        # 扩展t以匹配空间维度，拼接到通道中
        t_expand = t.view(-1, 1, 1, 1).expand(x.shape[0], 1, x.shape[2], x.shape[3])
        x_t = torch.cat([x, t_expand], dim=1)
        # 预测加在图像上的噪声 (noise residual)
        return self.conv(x_t)

class FlowLikelihoodEstimator(nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        # Normalizing Flow 模拟结构，用于算密度
        self.flow_layer = nn.Conv2d(channels, channels, 1) 
        
    def forward(self, x):
        # 模拟 Flow 前向传播，输出高斯隐变量 z 和 雅可比对数行列式 log_det
        z = self.flow_layer(x)
        log_det = torch.zeros(x.shape[0], device=x.device) 
        log_p_z = -0.5 * torch.sum(z**2, dim=1, keepdim=True) 
        log_likelihood = log_p_z + log_det.view(-1, 1, 1, 1)
        return log_likelihood

class PerStepGuidedDiffFlowAD(nn.Module):
    def __init__(self, channels=3, guidance_scale=0.1, n_steps=50):
        super().__init__()
        self.unet = DiffUNet(channels)
        self.flow_model = FlowLikelihoodEstimator(channels)
        self.guidance_scale = guidance_scale
        self.n_steps = n_steps
        
    def forward(self, x_orig):
        """
        前向推理（测试阶段定位异常）
        x_orig: [B, C, H, W] 待测试原图
        """
        batch_size = x_orig.shape[0]
        device = x_orig.device
        
        # --- 步骤1：前向加噪 ---
        # 加入微量或中等噪声，抹去高频的异常细节
        t_start = torch.tensor([self.n_steps], device=device, dtype=torch.float32)
        noise = torch.randn_like(x_orig)
        # 简化的线性加噪
        x_t = x_orig + noise * 0.5 
        
        # 累积异常偏差图 (Accumulated Anomaly Guidance)
        anomaly_score_map = torch.zeros(batch_size, 1, x_orig.shape[2], x_orig.shape[3], device=device)
        
        # --- 步骤2：反向联合去噪 (Flow 引导 Diffusion) ---
        for i in reversed(range(self.n_steps)):
            t = torch.tensor([i], device=device, dtype=torch.float32)
            
            # --- 2.1 获取 Flow 的引力梯度 ---
            # 必须使得输入 Tensor 持有梯度
            x_t.requires_grad_(True)
            log_p_xt = self.flow_model(x_t)
            
            # 对 log_likelihood 求总和标量后，向输入 x_t 求偏导
            log_p_xt_sum = log_p_xt.sum()
            grad_flow = torch.autograd.grad(log_p_xt_sum, x_t)[0] 
            
            # 及时detach x_t 避免计算图无限增长
            x_t = x_t.detach()
            
            # --- 2.2 获取 Diffusion 预测的基础去噪方向 ---
            pred_noise = self.unet(x_t, t)
            
            # --- 2.3 融合：步进更新 ---
            # 经典的去噪步骤 (这里用类似 DDIM 的简化后退)
            dt = 1.0 / self.n_steps
            # 基础更新 (走向 x_0)
            dx_diffusion = - pred_noise * dt 
            # Flow引导更新 (走向密集的正常工业件分布)
            dx_flow = self.guidance_scale * grad_flow * dt
            
            x_t = x_t + dx_diffusion + dx_flow
            
            # --- 2.4 异常得分累积 ---
            # 核心思想：正常区域，Diffusion就能去噪好，Flow给的修偏梯度很小。
            # 异常区域，Flow觉得非常“碍眼”，使得 grad_flow 模长在这里特别大。
            # 我们累积 Flow 在这整个去噪过程中，做出的“纠正努力强度”来作为异常得分。
            anomaly_score_map += torch.norm(grad_flow, dim=1, keepdim=True) * dt
        
        # 此时的 x_t 是重建完毕的近似正常图像，也可以额外叠加重建误差
        reconstruction_error = torch.mean((x_orig - x_t)**2, dim=1, keepdim=True)
        final_anomaly_map = anomaly_score_map + reconstruction_error
        
        return final_anomaly_map, x_t
