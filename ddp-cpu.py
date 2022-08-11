# This example shows how to initialize parameters of a module.  For details,
# please refer to https://stackoverflow.com/a/49433937/724872.

import torch
import sys
import os

WORLD_SIZE=2

def hello(rank: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    torch.distributed.init_process_group(backend='gloo', rank=rank, world_size=WORLD_SIZE)
    
    fc = torch.nn.Linear(in_features=1, out_features=1, bias=False, device="cpu")

    v = 1.0 * rank + 2.0
    fc.weight.data.fill_(v)
    assert torch.equal(fc.weight.data, torch.Tensor([[v]]))

    model = torch.nn.parallel.DistributedDataParallel(fc)
    # print(f"Rank {rank}: {fc.weight}\n")  # DDP unifies parameter values over processes.

    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    loss = torch.nn.MSELoss()

    for x in range(100):
        opt.zero_grad()
        a = torch.Tensor([[float(x % 10)]])
        b = torch.Tensor([[a * 3.14]])
        l = loss(model(a), b)
        l.backward()
        opt.step()

    assert torch.equal(fc.weight.data, torch.Tensor([[3.14]]))
    print(f"Rank {rank}: estimated {fc.weight.data}")
    sys.stdout.flush()  # must-to-have, or prints nothing.

if __name__ == '__main__':
    torch.multiprocessing.spawn(hello, nprocs=WORLD_SIZE)
