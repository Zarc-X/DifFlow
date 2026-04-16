import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace

from .components.ddpm_unet import StandardDDPMUNet
from .components.flow.flow_models import build_msflow_model

class DiffusionFlowAD(nn.Module):
    def __init__(self, channels=3, n_steps=400, freeze_unet=False):
        super().__init__()
        
        self.unet = StandardDDPMUNet(
            in_channels=channels, 
            out_channels=channels, 
            model_channels=64
        ).unet  
        
        self.freeze_unet = freeze_unet
        if self.freeze_unet:
            for param in self.unet.parameters():
                param.requires_grad = False
            self.unet.eval()

        self.n_steps = n_steps
        betas = torch.linspace(1e-4, 0.02, self.n_steps)
        alphas = 1.0 - betas
        self.register_buffer('alphas_cumprod', torch.cumprod(alphas, dim=0))
        
        self.t_stages = [50, 150, 250] 
        self.c_feats = [256, 256, 256] 
        
        c = SimpleNamespace(
            c_conds=[0, 0, 0],
            parallel_blocks=[8, 8, 8],  
            clamp_alpha=1.9
        )
        
        parallel_flows, _ = build_msflow_model(c, self.c_feats)
        self.parallel_flows = nn.ModuleList(parallel_flows)
        # 两阶段训练分界点：前期训练UNet去噪，后期冻结UNet仅训练Flow。
        self.warmup_epochs = 80

    def _sample_noise(self, x, deterministic=False, seed=0):
        if deterministic:
            gen = torch.Generator(device=x.device)
            gen.manual_seed(seed)
            return torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=gen)
        return torch.randn_like(x)
        
    def _extract_features(self, x, deterministic_noise=False):
        if self.freeze_unet:
            self.unet.eval()
            
        feature_maps = []
        for t_val in self.t_stages:
            t = torch.full((x.shape[0],), t_val, device=x.device).long()
            # Eval阶段使用固定随机种子生成的高斯噪声，既可复现又保持分布一致。
            noise = self._sample_noise(x, deterministic=deterministic_noise, seed=int(t_val))
            alpha_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
            x_noisy = torch.sqrt(alpha_cumprod_t) * x + torch.sqrt(1 - alpha_cumprod_t) * noise

            from .components.sdas.model_utils import timestep_embedding
            emb = self.unet.time_embed(timestep_embedding(t, self.unet.model_channels))
            h = x_noisy.type(self.unet.dtype)
            
            with torch.set_grad_enabled(not self.freeze_unet):
                for i, module in enumerate(self.unet.input_blocks):
                    h = module(h, emb)
                    if i == 8: 
                        feature_maps.append(h)
                        break 
        return feature_maps

    def forward_train(self, x_orig, epoch=None):
        device = x_orig.device
        batch_size = x_orig.shape[0]
        warmup_epochs = self.warmup_epochs
        train_unet = (not self.freeze_unet) and (epoch is None or epoch <= warmup_epochs)
        train_flow = self.freeze_unet or (epoch is None or epoch > warmup_epochs)
        
        # --- 1. 训练 DDPM U-Net (如果未冻结) ---
        loss_diff = torch.tensor(0.0, device=device, requires_grad=True)
        if train_unet:
            t = torch.randint(0, self.n_steps, (batch_size,), device=device).long()
            noise = self._sample_noise(x_orig, deterministic=False)
            alpha_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
            x_noisy = torch.sqrt(alpha_cumprod_t) * x_orig + torch.sqrt(1 - alpha_cumprod_t) * noise
            
            # 由于 StandardDDPMUNet 取出了 unet，直接通过 call() 全向计算
            pred_noise = self.unet(x_noisy.type(self.unet.dtype), t.float())
            loss_diff = F.mse_loss(pred_noise, noise)

        # --- 2. 训练 Flow 网络 ---
        loss_flow = torch.tensor(0.0, device=device)

        if train_flow:
            # 第二阶段冻结UNet统计行为，防止Flow训练期特征继续漂移。
            if not self.freeze_unet:
                self.unet.eval()

            with torch.no_grad():
                feats = self._extract_features(x_orig, deterministic_noise=False)

            loss_flow = 0.0
            zs = []
            for i, (p_flow, feat) in enumerate(zip(self.parallel_flows, feats)):
                # 【核心】：使用 detach() 截取特征！这正是 Idea 7 避免 Flow NLL 回传摧毁 U-Net 的灵魂。
                z_i, jac_i = p_flow(feat.detach())
                if isinstance(z_i, list) or isinstance(z_i, tuple): z_i = z_i[0]
                prior_logprob = -0.5 * torch.sum(z_i**2, dim=[1, 2, 3])
                log_likelihood = prior_logprob + jac_i
                loss_flow += -torch.mean(log_likelihood) / z_i[0].numel()
                zs.append(z_i)
            
        return loss_diff, loss_flow

    def forward_test(self, x_orig):
        feats = self._extract_features(x_orig, deterministic_noise=True)
        zs = []
        for i, p_flow in enumerate(self.parallel_flows):
            z_i, _ = p_flow(feats[i])
            if isinstance(z_i, tuple) or isinstance(z_i, list): z_i = z_i[0]
            zs.append(z_i)
        
        anomaly_map = None
        for z_i in zs:
            local_map = torch.mean(z_i**2, dim=1, keepdim=True)
            rescaled_map = F.interpolate(local_map, size=x_orig.shape[2:], mode='bilinear', align_corners=False)
            if anomaly_map is None:
                anomaly_map = rescaled_map
            else:
                anomaly_map += rescaled_map
                
        # 增加 U-Net 提取后完整去噪图像(recon_img) 用于可视化
        # 选取一个中间加噪时间步 t=100
        t_recon = torch.full((x_orig.shape[0],), 100, device=x_orig.device).long()
        noise = self._sample_noise(x_orig, deterministic=True, seed=100)
        alpha_cumprod_t = self.alphas_cumprod[t_recon].view(-1, 1, 1, 1)
        x_noisy = torch.sqrt(alpha_cumprod_t) * x_orig + torch.sqrt(1 - alpha_cumprod_t) * noise
        
        with torch.no_grad():
            # 临时走一个完整的前向来得到单步去噪图
            pred_noise = self.unet(x_noisy, t_recon.float())
            recon_img = (x_noisy - torch.sqrt(1 - alpha_cumprod_t) * pred_noise) / torch.sqrt(alpha_cumprod_t)
            
        return anomaly_map, recon_img
