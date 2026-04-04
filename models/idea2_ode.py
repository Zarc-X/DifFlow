import torch
import torch.nn as nn
from .components.ddpm_unet import StandardDDPMUNet

class ProbabilityFlowODE_AD(nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        # 拟合速度场 v(x, t) 的网络，使用标准的DDPM UNet
        self.v_model = StandardDDPMUNet(in_channels=channels, out_channels=channels, model_channels=64)
        
    def get_divergence(self, x, t):
        """
        利用 Hutchinson trace estimator 计算速度场的散度。
        """
        x.requires_grad_(True)
        v = self.v_model(x, t)
        
        # Hutchinson Trace Estimator: div(v) ≈ E[e^T * \nabla_x (v) * e]
        e = torch.randn_like(x).sign() # Rademacher vector
        
        v_dot_e = torch.sum(v * e)
        
        # 求特征雅可比矩阵向量积
        grad_v_e = torch.autograd.grad(v_dot_e, x, create_graph=False)[0]
        
        # 针对每个空间位置计算通道维度的求和，作为局部散度的近似
        div = torch.sum(grad_v_e * e, dim=1, keepdim=True)
        
        return v.detach(), div.detach()

    def forward_train(self, x):
        """
        前向训练过程 (基于 Flow Matching)
        """
        batch_size = x.shape[0]
        t_val = torch.rand(batch_size, device=x.device).float()
        noise = torch.randn_like(x)
        
        # 【更正】采用标准 Flow Matching: t=0是纯噪声，t=1是清晰分布（x_1）
        # x_t = (1-t) * noise + t * x
        # 因为 v网络 要预测的是 dx/dt 速度场。
        # 速度场 target_v = x - noise
        
        t_channel = t_val.view(-1, 1, 1, 1)
        x_t = (1 - t_channel) * noise + t_channel * x 
        target_v = x - noise
        
        v = self.v_model(x_t, t_val)
        loss = torch.nn.functional.mse_loss(v, target_v)
        
        return loss, torch.tensor(0.0, device=x.device, requires_grad=True)

    def forward_test(self, x_0):
        """
        前向推理过程（沿速度场逆流 ODE 积分，求解原图对数似然）
        """
        # 测试时我们需要关闭内部权重的大量内存需求
        prev_grad_states = [p.requires_grad for p in self.parameters()]
        for p in self.parameters():
            p.requires_grad = False

        try:
            batch_size = x_0.shape[0]
            
            # 【重要修复】积分方向与极性更正
            # 我们要由 x_0 倒退回高斯纯噪声计算先验，因此状态起点应当是 t=1
            x_t = x_0.clone()
            delta_log_p = torch.zeros(batch_size, 1, x_0.shape[2], x_0.shape[3], device=x_0.device)
            
            n_steps = 20
            dt = 1.0 / n_steps
            
            # 时间沿流形倒流：从 t=1 倒退到 t=0
            # dt 取正值，我们做减法回退
            for i in reversed(range(1, n_steps + 1)):
                t_val = torch.tensor([i * dt] * batch_size, device=x_0.device, dtype=torch.float32)
                
                # 局部计算图隔离，仅求相对于当前 x_t 的散度偏导
                x_t = x_t.detach()
                x_t.requires_grad_(True)
                
                with torch.enable_grad():
                    v, div = self.get_divergence(x_t, t_val)
                
                # 时间逆流回退： x_{t-dt} = x_t - v * dt
                # 对数似然的逆常微分推导：log p(x_0) = log p(x_{t_{0}}) + ∫ div(v) dt
                x_t = (x_t - v * dt).detach()
                delta_log_p = (delta_log_p + div * dt).detach()
                
            # 到达 t=0 时，此时 x_t 应退化服从标准高斯 N(0, I)
            prior_log_p = -0.5 * torch.sum(x_t**2, dim=1, keepdim=True)
            
            # 完整对数似然 = 先验似然 + 从先验演变过来的积分偏差
            log_likelihood = prior_log_p + delta_log_p
            
            # 取负数为异常分数（越不可能出现的东西越异常）
            anomaly_map = -log_likelihood
            
            return anomaly_map, x_t
        finally:
            for p, state in zip(self.parameters(), prev_grad_states):
                p.requires_grad = state
