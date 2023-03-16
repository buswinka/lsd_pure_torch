import torch
import torch.multiprocessing as mp

from bism.models import get_constructor
from bism.models.lsd import LSDModel

from src.train.distributed import train
from src.train.setup import setup_process, cleanup, find_free_port

from functools import partial

torch.set_float32_matmul_precision('high')

dims = [32, 64, 128, 64, 32]

model_constructor = get_constructor('unext', spatial_dim=3)  # gets the model from a name...
backbone = model_constructor(in_channels=1, out_channels=10, dims=dims)
model = LSDModel(backbone)


test_input = torch.rand((1, 1, 300, 300, 20))
out = model(test_input)

hyperparams = {
    'model': 'unext',
    'depths': '[2,2,2,2,2]',
    'dims': str(dims),
}

if __name__ == '__main__':

    port = find_free_port()
    world_size = 2
    mp.spawn(train, args=(port, world_size, model, hyperparams), nprocs=world_size, join=True)
