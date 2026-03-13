import torch
import torch.nn as nn

"""
思路二：速度场与常微分方程 (ODE) 的深度融合 (Probability Flow ODE / Flow Matching)

【架构思想】：底层数学视角的统一（理论级融合）。
这代表了最近（如 Rectified Flow, Score SDE）最前沿的统一视角。
1. 抛弃了传统的马尔可夫噪声链（多步加噪/去噪）。
2. 让网络去学习一个由噪声 x_T 到真实数据 x_0 映射的 “速度场” (Velocity Field): dx/dt = v(x, t)。
3. 给定一张测试图片，我们可以让它沿着这个通过训练学到的速度场作 ODE 逆向积分 (积分到纯噪声)。
4. 在积分的同时，利用瞬时变量替换定理 (Instantaneous Change of Variables) 算出精确对数似然 (Exact Log-Likelihood)。
5. 正常正常样本平滑退化为标准高斯，异常样本在积分解算时产生极低的似然或轨迹偏移。
"""

class VelocityFieldNet(nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        # 拟合速度场 v(x, t) 的网络，通常也是一个 U-Net 架构
        # 输入维度是 x(t) 和 时间标量 t
        self.net = nn.Sequential(
            nn.Conv2d(channels + 1, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, channels, 3, padding=1)
        )
        
    def forward(self, x, t):
        # 将标量 t 扩展并拼接在通道维度上
        t_channel = t.view(-1, 1, 1, 1).expand(x.size(0), 1, x.size(2), x.size(3))
        x_t = torch.cat([x, t_channel], dim=1)
        # 输出此时的 速度场向量
        dx_dt = self.net(x_t)
        return dx_dt

class ProbabilityFlowODE_AD(nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        self.v_model = VelocityFieldNet(channels)
        
    def get_divergence(self, x, t):
        """
        利用 Hutchinson trace estimator 或自动微分计算速度场的散度(Divergence)。
        这是计算连流流模型 (Continuous Normalizing Flow) 概率密度的核心。
        d(log p(x(t))) / dt = - div(v) = - Tr(dv/dx)
        """
        # 注意：实际应用中由于全量雅可比矩阵求迹太慢，通常采用 Hutchinson 随机估算
        # 此处使用伪代码演示概念
        x.requires_grad_(True)
        v = self.v_model(x, t)
        
        # 伪散度计算 (仅作原理展示)
        div = torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device)
        return v, div

    def forward_train(self, x):
        """
        前向训练过程 (模拟 Flow Matching / Score Matching 的加噪和回归)
        为了兼容 train.py 的双损失输出，第二个loss占位0
        """
        # 简化版拟合速度场目标
        t_val = torch.rand(x.shape[0], 1, 1, 1, device=x.device)
        noise = torch.randn_like(x)
        x_t = t_val * x + (1 - t_val) * noise
        target_v = x - noise
        
        v = self.v_model(x_t, t_val.view(-1))
        loss = torch.nn.functional.mse_loss(v, target_v)
        return loss, torch.tensor(0.0, device=x.device)

    def forward_test(self, x_0):
        """
        前向推理过程（测试阶段计算 Exact Log Likelihood）
        使用数值 ODE Solver (如 Euler 法或 Runge-Kutta) 对速度场进行向后积分
        """
        batch_size = x_0.shape[0]
        # 初始化积分状态：t=0 时刻的数据为目标图像 x_0
        x_t = x_0
        # 初始概率密度增量为 0
        delta_log_p = torch.zeros(batch_size, 1, x_0.shape[2], x_0.shape[3], device=x_0.device)
        
        # 定义积分步数和时间范围 t ∈ [0, 1]
        n_steps = 20
        dt = 1.0 / n_steps
        
        # 执行常微分方程求解 (ODE Integration)
        for i in range(n_steps):
            t = torch.tensor([i * dt], device=x_0.device)
            # 计算当前时刻的速度场 v(x, t) 和 散度 div(x, t)
            v, div = self.get_divergence(x_t, t)
            
            # 使用简单的欧拉法 (Euler Method) 更新状态 x 和 log p
            x_t = x_t + v * dt                    # 在形态流形上向噪声方向走
            delta_log_p = delta_log_p - div * dt  # 累加沿途概率密度的变化量
            
        # 积分到 t=1 时，x_t 已经被映射为先验噪声分布 (通常是标准高斯分布 N(0, I))
        # 计算该噪声在标准高斯中的基础概率 log P_1(x_1)
        prior_log_p = -0.5 * torch.sum(x_t**2, dim=1, keepdim=True)
        
        # 源图像 x_0 的精确对数似然 = 终点先验概率 + 轨迹散度积分
        log_likelihood = prior_log_p + delta_log_p
        
        # 将负对数似然用作异常分数地图
        anomaly_map = -log_likelihood
        
        return anomaly_map, x_t
