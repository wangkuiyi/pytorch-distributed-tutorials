# This is basically the example program https://pytorch.org/docs/stable/pipeline.html#pipe-apis-in-pytorch
# tailored for running CPU.  So, we can run it on macOS.

import torch
from torch.distributed.pipeline.sync import Pipe
import os

# Need to initialize RPC framework first.
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"
torch.distributed.rpc.init_rpc("worker", rank=0, world_size=1)

# Build pipe.
fc1 = torch.nn.Linear(16, 8)
fc2 = torch.nn.Linear(8, 4)
model = torch.nn.Sequential(fc1, fc2)

model = Pipe(model, chunks=8)
input = torch.rand(16, 16)
output_rref = model(input)
assert output_rref.to_here().size() == torch.Size([16, 4])
