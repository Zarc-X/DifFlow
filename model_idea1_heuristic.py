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
    def __init__(self, channels=3, base_dim=64):
        super().__init__()
        # 简单可运行的U-Net用于演示和跑库验证。实际工业落地请换成DDPM中的标准UNet架构
        self.enc1 = nn.Conv2d(channels + 1, base_dim, 3, padding=1)
        self.enc2 = nn.Conv2d(base_dim, base_dim*2, 3, stride=2, padding=1)
        self.dec2 = nn.ConvTranspose2d(base_dim*2, base_dim, 4, stride=2, padding=1)
        self.dec1 = nn.Conv2d(base_dim * 2, channels, 3, padding=1)

    def forward(self, x, t):
        # 将 t 扩展拼接进入特征
        t_expand = t.view(-1, 1, 1, 1).expand(x.shape[0], 1, x.shape[2], x.shape[3])
        x_in = torch.cat([x, t_expand], dim=1)
        
        e1 = F.relu(self.enc1(x_in))
        e2 = F.relu(self.enc2(e1))
        d2 = F.relu(self.dec2(e2))
        
        d1 = self.dec1(torch.cat([e1, d2], dim=1))
        return d1

class FlowLikelihoodEstimator(nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        # 简单模拟 Normalizing Flow (RealNVP/Glow 等)。工业应用请使用 msflow 中的 Freia 块。
        self.flow_layer = nn.Sequential(
            nn.Conv2d(channels, 64, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, channels, 1)
        )
        
    def forward(self, x):
        # 伪装 Flow 的对数似然计算：假设隐变量z符合标准正态分布，并粗略近似雅可比矩阵
        z = self.flow_layer(x)
        # 用常数或近似项模拟log_det_jacobian
        log_det = torch.zeros(x.shape[0], device=x.device) 
        log_p_z = -0.5 * torch.sum(z**2, dim=1, keepdim=True) 
        log_likelihood = log_p_z + log_det.view(-1, 1, 1, 1)
        return log_likelihood

class PerStepGuidedDiffFlowAD(nn.Module):
    def __init__(self, channels=3, guidance_scale=0.1, n_steps=50):
        super().__init__()
        self.unet = DiffUNet(channels=channels)
        self.flow_model = FlowLikelihoodEstimator(channels=channels)
        self.guidance_scale = guidance_scale
        self.n_steps = n_steps
        
    def forward_train(self, x_0):
        """
        训练阶段：分别计算 Diffusion 的去噪MSE损失，以及 Flow 的对数似然损失
        """
        device = x_0.device
        batch_size = x_0.shape[0]
        
        # ------------------
        # 1. Diffusion Loss
        # ------------------
        # 随机采样时间步 [0, n_steps-1]
        t = torch.randint(0, self.n_steps, (batch_size,), device=device).float()
        noise = torch.randn_like(x_0)
        
        # 为了演示，此处采用简化的纯线性加噪（真实场景请使用 DDPM 的 alpha/beta 调度机制）
        alpha_t = 1.0 - (t / self.n_steps)
        alpha_t = alpha_t.view(-1, 1, 1, 1)
        x_noisy = alpha_t * x_0 + (1 - alpha_t) * noise
        
        pred_noise = self.unet(x_noisy, t)
        loss_diff = F.mse_loss(pred_noise, noise)
        
        # ------------------
        # 2. Flow Loss
        # ------------------
        # 训练 Flow 拟合原图(正常纯净样本)的概率密度 (Max Likelihood 视角)
        log_p = self.flow_model(x_0)
        loss_flow = -torch.mean(log_p)  # 最小化负对数似然
        
        return loss_diff, loss_flow

    def forward_test(self, x_orig):
        """
        测试阶段：推理获取 Anomaly Score Map (利用Flow引力梯度引导)
        """
        batch_size = x_orig.shape[0]
        device = x_orig.device
        
        t_start = torch.tensor([self.n_steps-1] * batch_size, device=device, dtype=torch.float32)
        noise = torch.randn_like(x_orig)
        x_t = x_orig * 0.5 + noise * 0.5 # 适度加噪破坏异常，不建议毁到全噪
        
        anomaly_score_map = torch.zeros(batch_size, 1, x_orig.shape[2], x_orig.shape[3], device=device)
        
        for i in reversed(range(self.n_steps)):
            t = torch.tensor([i] * batch_size, device=device, dtype=torch.float32)
            
            # --- Flow 引力梯度引导 ---
            x_t.requires_grad_(True)
            log_p_xt = self.flow_model(x_t)
            log_p_xt_sum = log_p_xt.sum()
            grad_flow = torch.autograd.grad(log_p_xt_sum, x_t)[0] 
            x_t = x_t.detach()
            
            # --- Diffusion 去噪预测 ---
            pred_noise = self.unet(x_t, t)
            
            # --- 更新融合 ---
            dt = 1.0 / self.n_steps
            dx_diffusion = - pred_noise * dt 
            dx_flow = self.guidance_scale * grad_flow * dt
            
            x_t = x_t + dx_diffusion + dx_flow
            
            # --- 累积异常得分 ---
            anomaly_score_map += torch.norm(grad_flow, dim=1, keepdim=True) * dt
            
        reconstruction_error = torch.mean((x_orig - x_t)**2, dim=1, keepdim=True)
        final_anomaly_map = anomaly_score_map + reconstruction_error
        
        return final_anomaly_map, x_t

