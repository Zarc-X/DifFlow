import torch
import FrEIA.framework as Ff
import FrEIA.modules as Fm
import torch.nn as nn
from models.components.flow.flow_models import subnet_conv_ln

flows = Ff.SequenceINN(64, 1, 1)
flows.append(Fm.AllInOneBlock, subnet_constructor=subnet_conv_ln, affine_clamping=1.9, global_affine_type='SOFTPLUS')

# input is 64x32x32
x = torch.randn(1, 64, 32, 32)
x.requires_grad_(True)
z, jac = flows(x)
print(z.shape)

z[0, 0, 16, 16].backward()
# check if gradient of x is non-zero ONLY around 16,16
print((x.grad.abs().sum(dim=1) > 0).nonzero())
