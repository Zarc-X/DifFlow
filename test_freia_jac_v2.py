import torch
import FrEIA.framework as Ff
import FrEIA.modules as Fm
from models.components.flow.flow_models import subnet_conv_ln

flows = Ff.SequenceINN(64, 1, 1)
flows.append(Fm.AllInOneBlock, subnet_constructor=subnet_conv_ln, affine_clamping=1.9, global_affine_type='SOFTPLUS')

# Force large scale:
for name, param in flows.named_parameters():
    if 'global_scale' in name:
        param.data.fill_(2.0)

x = torch.randn(1, 64, 32, 32)
z, jac = flows(x)
print("jac:", jac)
