# This example shows how to use the package torch.multiprocessing
# to spawn Python processes.  This is a common operation that we
# will use in the rest examples of distributed training.

import torch
import sys

def hello(rank: int):
    print(f"hello, rank {rank}!")
    sys.stdout.flush()

if __name__ == '__main__':
    torch.multiprocessing.spawn(hello, nprocs=2)
