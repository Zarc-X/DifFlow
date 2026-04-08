import torch
from models.idea6_decoupled import DecoupledDiffFlowAD
model = DecoupledDiffFlowAD()
x = torch.randn(2, 3, 256, 256)
model.eval()
with torch.no_grad():
    feats = model._extract_features(x)
    print("Feat shapes:", [f.shape for f in feats])
    zs = []
    for i, p_flow in enumerate(model.parallel_flows):
        z_i, _ = p_flow(feats[i])
        print(f"z_{i} shape:", z_i[0].shape)
        
