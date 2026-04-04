import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .components.ddpm_unet import StandardDDPMUNet
from .components.freia_flow import NormalizingFlowDensityEstimator

class SimpleGuidedDiffFlowAD(nn.Module):
    def __init__(self, channels=3, guidance_scale=0.01, n_steps=50):
        super().__init__()
        # 直接复用项目中现成的基础 UNet 和 Flow
        self.unet = StandardDDPMUNet(in_channels=channels, out_channels=channels, model_channels=64)
        self.flow_model = NormalizingFlowDensityEstimator(in_channels=channels)
        self.guidance_scale = guidance_scale
        self.n_steps = n_steps
        
        # --- 【修复1】构建基于数学推导的标准 DDPM (Diffusion) 调度表 ---
        beta_start = 0.0001
        beta_end = 0.02
        betas = torch.linspace(beta_start, beta_end, n_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        
    def forward_train(self, x_0):
        device = x_0.device
        batch_size = x_0.shape[0]
        
        # --- 1. Diffusion Loss ---
        # 随机抽取时间步
        t = torch.randint(0, self.n_steps, (batch_size,), device=device).long()
        noise = torch.randn_like(x_0)
        
        # 标准前向加噪公式：x_t = sqrt(alpha_bar)*x_0 + sqrt(1-alpha_bar)*noise
        a_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        x_noisy = torch.sqrt(a_cumprod_t) * x_0 + torch.sqrt(1.0 - a_cumprod_t) * noise
        
        pred_noise = self.unet(x_noisy, t.float())
        loss_diff = F.mse_loss(pred_noise, noise)
        
        # --- 2. Flow Loss ---
        z, log_jac_det = self.flow_model(x_0)
        
        # 引入高斯先验，计算对数似然 (N(0, I) 下的截断似然)
        prior_logprob = -0.5 * torch.sum(z**2, dim=[1,2,3]) - 0.5 * math.log(2*math.pi) * z[0].numel()
        log_likelihood = prior_logprob + log_jac_det
        
        # 【修复2】除以维度均摊，防止负数无限大破坏梯度传递平衡
        loss_flow = -torch.mean(log_likelihood) / z[0].numel()
        
        return loss_diff, loss_flow

    def forward_test(self, x_orig):
        # 临时关闭模型所有权重的梯度追踪
        prev_grad_states = [p.requires_grad for p in self.parameters()]
        for p in self.parameters():
            p.requires_grad = False

        try:
            batch_size = x_orig.shape[0]
            device = x_orig.device
            
            # --- 【修复3】正确配置测试起点 ---
            # 测试异常检测时不能从纯噪声(t=100%)开始，否则完全失去原图结构比对参照。
            # 通常从 t≈50% (比如这25步的地方) 开始进行重构加噪
            start_step = int(self.n_steps * 0.5) 
            
            a_cumprod_start = self.alphas_cumprod[start_step].view(-1, 1, 1, 1)
            noise = torch.randn_like(x_orig)
            x_t = torch.sqrt(a_cumprod_start) * x_orig + torch.sqrt(1.0 - a_cumprod_start) * noise
            
            anomaly_score_map = torch.zeros(batch_size, 1, x_orig.shape[2], x_orig.shape[3], device=device)
            
            # 使用 DDIM 的方式往回逐步确定性去噪
            for i in reversed(range(start_step)):
                t_tensor = torch.tensor([i] * batch_size, device=device, dtype=torch.long)
                t_float = t_tensor.float()
                
                a_cumprod_t = self.alphas_cumprod[i].view(-1, 1, 1, 1)
                a_cumprod_t_prev = self.alphas_cumprod[i-1].view(-1, 1, 1, 1) if i > 0 else torch.tensor(1.0, device=device).view(-1, 1, 1, 1)
                
                # --- Diffusion: 预测当前步噪声 ---
                with torch.no_grad():
                    pred_noise = self.unet(x_t, t_float)
                
                # --- 【修复4】数学更正的引导求导机制 ---
                # 用测测到的噪声还原出此时认为的”干净图 x_0“(pred_x0)
                # 只有针对 pred_x0 使用 Flow 才是有意义也是最严谨的
                x_t_detached = x_t.detach().requires_grad_(True)
                pred_x0_grad = (x_t_detached - torch.sqrt(1.0 - a_cumprod_t) * pred_noise) / torch.sqrt(a_cumprod_t)
                pred_x0_grad = torch.clamp(pred_x0_grad, -1.0, 1.0)
                
                with torch.enable_grad():
                    z_curr, log_jac_det = self.flow_model(pred_x0_grad)
                    
                    prior_logprob = -0.5 * torch.sum(z_curr**2, dim=[1,2,3])
                    log_likelihood_sum = (prior_logprob + log_jac_det).sum() / z_curr[0].numel()
                    
                    # 针对 x_t 求导拿到指导去噪方向的正向梯度
                    grad_flow = torch.autograd.grad(log_likelihood_sum, x_t_detached, create_graph=False)[0]
                
                # 累加 Flow 暴露出的异常抗拒梯度值，作为定位参考一部分
                anomaly_score_map += torch.norm(grad_flow, p=2, dim=1, keepdim=True)
                
                # --- DDIM 标准积分更新 ---
                # 在没有噪声重新注入的前提下用 DDIM 从 pred_x0 逼近 x_{t-1}
                dir_xt = torch.sqrt(1.0 - a_cumprod_t_prev) * pred_noise
                x_t_prev = torch.sqrt(a_cumprod_t_prev) * pred_x0_grad.detach() + dir_xt
                
                # 加上 Flow Guidance (引导其靠向正常数据流流形分布)
                x_t = x_t_prev + self.guidance_scale * grad_flow
                x_t = x_t.detach()
            
            # --- 【修复5】计算重构误差 ---
            # 原图减去回退后的图像。如果包含异常点，模型无法将其重建为原样。
            reconstruction_error = torch.mean((x_orig - x_t)**2, dim=1, keepdim=True)
            
            # 整合特征偏离度与重构偏差
            final_anomaly_map = reconstruction_error + 0.1 * anomaly_score_map
            
            return final_anomaly_map, x_t

        finally:
            for p, state in zip(self.parameters(), prev_grad_states):
                p.requires_grad = state
