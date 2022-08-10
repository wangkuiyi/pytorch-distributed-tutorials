# This example shows how to initialize parameters of a module.  For details,
# please refer to https://stackoverflow.com/a/49433937/724872.

import torch

fc = torch.nn.Linear(in_features=2, out_features=2, bias=False, device="cpu")
torch.nn.init.eye_(fc.weight)

a = torch.Tensor([0.1, 0.2])
b = fc(a)
assert torch.equal(a, b)  # Because fc.weigth is an identity matrix
