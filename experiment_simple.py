import torch
from models.idea4_simple import SimpleGuidedDiffFlowAD

# 这是一个快速验证小脚本，确保刚才写的代码不报错并且没有外部依赖约束。
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing on {device}...")
    
    # 实体化轻量级网络
    model = SimpleGuidedDiffFlowAD(channels=3, guidance_scale=0.1, n_steps=20).to(device)
    
    # 构造假图片 B, C, H, W
    dummy_img = torch.randn(2, 3, 64, 64).to(device)
    
    # 测试训练向前推导通道
    loss_diff, loss_flow = model.forward_train(dummy_img)
    print(f"Train passes! Diffusion Loss: {loss_diff.item():.4f}, Flow Loss: {loss_flow.item():.4f}")
    
    # 测试引力推断通道
    anomaly_map, restored_img = model.forward_test(dummy_img)
    print(f"Infer passes! Anomaly Map shape: {anomaly_map.shape}, Max Score: {anomaly_map.max().item():.4f}")
