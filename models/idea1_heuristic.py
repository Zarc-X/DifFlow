import torch
import torch.nn as nn
import torch.nn.functional as F
from .components.ddpm_unet import StandardDDPMUNet
from .components.freia_flow import NormalizingFlowDensityEstimator
import math

class FlowLikelihoodEstimatorWrapper(nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        self.flow_layer = NormalizingFlowDensityEstimator(in_channels=channels, n_blocks=4)
        
    def forward(self, x):
        z, log_det = self.flow_layer(x)
        # prior log likelihood N(0, I)
        log_p_z = -0.5 * torch.sum(z**2, dim=1, keepdim=True) - 0.5 * math.log(2 * math.pi) * z.shape[1]
        
        # log_det in FrEIA might be a 1D tensor [B], we need to shape it to [B, 1, 1, 1] if required
        if log_det.dim() == 1:
            log_det = log_det.view(-1, 1, 1, 1)
        # Summing spatial dimensions of log_det if necessary, etc.
        # But actually log_det is usually the sum over all dimensions per batch item. 
        # Wait, FrEIA jac is [B]. So log_p_z should also be summed over all dims per batch.
        log_p_z_sum = log_p_z.view(x.shape[0], -1).sum(dim=1, keepdim=True)
        
        log_likelihood_total = log_p_z_sum + log_det.view(-1, 1)
        
        # We need a log_likelihood per pixel for local guidance? Idea 1 needs "∇_x log P_flow(x_t)"
        # This will just compute the gradient of the TOTAL likelihood w.r.t x, which gives a gradient map of shape x.
        return log_likelihood_total

class PerStepGuidedDiffFlowAD(nn.Module):
    def __init__(self, channels=3, guidance_scale=0.1, n_steps=50):
        super().__init__()
        self.unet = StandardDDPMUNet(in_channels=channels, out_channels=channels, model_channels=64)
        self.flow_model = FlowLikelihoodEstimatorWrapper(channels=channels)
        self.guidance_scale = guidance_scale
        self.n_steps = n_steps
        
    def forward_train(self, x_0):
        device = x_0.device
        batch_size = x_0.shape[0]
        
        # 1. Diffusion Loss
        t = torch.randint(0, self.n_steps, (batch_size,), device=device).float()
        noise = torch.randn_like(x_0)
        
        alpha_t = 1.0 - (t / self.n_steps)
        alpha_t = alpha_t.view(-1, 1, 1, 1)
        x_noisy = alpha_t * x_0 + (1 - alpha_t) * noise
        
        pred_noise = self.unet(x_noisy, t)
        loss_diff = F.mse_loss(pred_noise, noise)
        
        # 2. Flow Loss
        log_p = self.flow_model(x_0)
        loss_flow = -torch.mean(log_p)
        
        return loss_diff, loss_flow

    def forward_test(self, x_orig):
        batch_size = x_orig.shape[0]
        device = x_orig.device
        
        t_start = torch.tensor([self.n_steps-1] * batch_size, device=device, dtype=torch.float32)
        noise = torch.randn_like(x_orig)
        x_t = x_orig * 0.5 + noise * 0.5
        
        anomaly_score_map = torch.zeros(batch_size, 1, x_orig.shape[2], x_orig.shape[3], device=device)
        
        for i in reversed(range(self.n_steps)):
            t = torch.tensor([i] * batch_size, device=device, dtype=torch.float32)
            
            # Flow Guidance
            x_t.requires_grad_(True)
            log_p_xt_sum = self.flow_model(x_t).sum()
            grad_flow = torch.autograd.grad(log_p_xt_sum, x_t)[0] 
            x_t = x_t.detach()
            
            # Diffusion pred
            pred_noise = self.unet(x_t, t)
            
            dt = 1.0 / self.n_steps
            dx_diffusion = - pred_noise * dt 
            dx_flow = self.guidance_scale * grad_flow * dt
            
            x_t = x_t + dx_diffusion + dx_flow
            
            anomaly_score_map += torch.norm(grad_flow, dim=1, keepdim=True) * dt
            
        reconstruction_error = torch.mean((x_orig - x_t)**2, dim=1, keepdim=True)
        final_anomaly_map = anomaly_score_map + reconstruction_error
        
        return final_anomaly_map, x_t

