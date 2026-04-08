import torch
from models.components.ddpm_unet import StandardDDPMUNet

model = StandardDDPMUNet()
x = torch.randn(1, 3, 256, 256)
t = torch.tensor([0.0])

features = {}
def get_hook(name):
    def hook(module, input, output):
        features[name] = output.shape
    return hook

for i in range(12):
    model.unet.input_blocks[i].register_forward_hook(get_hook(i))

with torch.no_grad():
    model(x, t)

for k, v in features.items():
    print(f"Block {k}: {v}")
