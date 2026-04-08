import torch
import FrEIA.framework as Ff
import FrEIA.modules as Fm
from models.components.flow.flow_models import subnet_conv_ln

flows = Ff.SequenceINN(64, 1, 1)
flows.append(Fm.AllInOneBlock, subnet_constructor=subnet_conv_ln, affine_clamping=1.9, global_affine_type='SOFTPLUS')

x = torch.randn(1, 64, 32, 32)
z, jac = flows(x)
print("x shape:", x.shape)
print("jac shape:", jac.shape)
print("jac value:", jac)
