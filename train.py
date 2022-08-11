# This example shows how to initialize parameters of a module.  For details,
# please refer to https://stackoverflow.com/a/49433937/724872.

import torch
import sys
import os

def hello():
    model = torch.nn.Linear(in_features=1, out_features=1, bias=False, device="cpu")
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    loss = torch.nn.MSELoss()

    for x in range(100):
        opt.zero_grad()
        a = torch.Tensor([[float(x % 10)]])
        b = torch.Tensor([[a * 3.14]])

        y = model(a)
        l = loss(y, b)
        print(f"y: {y}, loss: {l}")
        l.backward()
        opt.step()

    print(f"Estimated {model.weight.data}")
    sys.stdout.flush()  # must-to-have, or prints nothing.

if __name__ == '__main__':
    hello()
