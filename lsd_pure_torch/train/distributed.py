import warnings
from functools import partial
from typing import Tuple, Callable, Dict
import os.path

from lsd_pure_torch.train.dataloader import dataset, MultiDataset, colate
from lsd_pure_torch.train.merged_transform import merged_transform_3D
from lsd_pure_torch.train.engine import engine
from lsd_pure_torch.train.setup import setup_process, cleanup, find_free_port

from torch import Tensor
import torch.nn as nn
import torch.optim.lr_scheduler
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from lion_pytorch import Lion

"""
ASSUMPTIONS: 
    - Model predicts spatially accurate vector strengths. Ie. 2 in a vector is 60*(0.085nm)
    - Vector Scale Factor turns 1,1,1 -> 60 60 12
    - Sigma is the spatial correction factor, which can vary between XYZ. Sigma represents distance in PX space
    - Closest Skeleton must take into account SPATIAL distance, not px distance

PLACES WHICH HANDLE ANISOTROPY
    - skoots.train.generate_skeletons.calculate_skeletons also takes some anisotropy into account...
    - vec2embed handles anisotropy
    - sigma in embed2prob handles anisotropy
    - Baked Skeleton: When calculating the optimal skeleton location, it 
    
CURRENT STRATEGY
    - Baked Skeleton Anisotropy (1, 1, 1) # should be 1, 1, 5 in TRAIN mode but (1, 1, 1) in eval
    - Vec2Embed (60, 60, 12)  # ratio:(1, 1, 5)
    - Sigma (20, 20, 4)  # ratio:(1, 1, 5)
    
    
NEW STRATEGY (Oct 21):
    - These models seem to like embedding to the current Z slice more than up or down. Adjust bkaed
    skeleon anisotropy to (1, 1, 15) and see what happens
    - set skeleton loss start at epoch 100 (from 500) for pretrained models...
    - skeleton overlap loss causes skeletons to be phat
    - EVAL anisotropy param for bake skeletons should also probably be huge (1. 1. 15)
"""

torch.manual_seed(101196)


def train(rank: str,
          port: str,
          world_size: int,
          model: nn.Module,
          hyperparams,
          train_dir: str = '/home/chris/Dropbox (Partners HealthCare)/trainMitochondriaSegmentation/data/unscaled/train',
          validation_dir: str = '/home/chris/Dropbox (Partners HealthCare)/trainMitochondriaSegmentation/data/unscaled/validate',
          ):
    setup_process(rank, world_size, port, backend='nccl')

    device = f'cuda:{rank}'

    model = model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model)

    _ = model(torch.rand((1, 1, 300, 300, 20), device=device))

    augmentations: Callable[[Dict[str, Tensor]], Dict[str, Tensor]] = partial(merged_transform_3D,
                                                                              device=device)

    # Training Dataset - MultiDataset[Mitochondria, Background]
    data = dataset(path=train_dir,
                   transforms=augmentations,
                   sample_per_image=32,
                   device=device,
                   pad_size=10).to(device)

    train_sampler = torch.utils.data.distributed.DistributedSampler(data)
    dataloader = DataLoader(data, num_workers=0, batch_size=1, sampler=train_sampler, collate_fn=colate)

    # Validation Dataset
    vl = dataset(path=validation_dir,
                 transforms=augmentations, device=device, sample_per_image=8,
                 pad_size=100).to(device)

    test_sampler = torch.utils.data.distributed.DistributedSampler(vl)
    valdiation_dataloader = DataLoader(vl, num_workers=0, batch_size=1, sampler=test_sampler,
                                       collate_fn=colate)

    torch.backends.cudnn.benchmark = True
    torch.autograd.profiler.profile = False
    torch.autograd.profiler.emit_nvtx(enabled=False)
    torch.autograd.set_detect_anomaly(False)

    # anisotropy is roughly (1, 1, 5)

    # The constants dict contains everything needed to replicate a training run.
    # will get serialized and saved.
    epochs = 10000
    constants = {
        'model': model,
        'lr': 1e-3,  # 5e-4 / 3,
        'wd': 1e-6,  # 1e-6 * 3,
        'optimizer': partial(torch.optim.AdamW, eps=1e-16), #partial(Lion, use_triton=False), #,
        'scheduler': partial(torch.optim.lr_scheduler.CosineAnnealingWarmRestarts, T_0=epochs+ 1),
        'loss_fn': torch.nn.MSELoss(), #tversky(alpha=0.25, beta=0.75, eps=1e-8, device=device),
        'epochs': epochs,
        'device': device,
        'train_data': dataloader,
        'val_data': valdiation_dataloader,
        'train_sampler': train_sampler,
        'test_sampler': test_sampler,
        'distributed': True,
        'mixed_precision': True,
        'rank': rank,
        'n_warmup': 10,
        'savepath': '/home/chris/Desktop/lsd_pure_torch/models',
    }

    writer = SummaryWriter() if rank == 0 else None
    if writer:
        print('SUMMARY WRITER LOG DIR: ', writer.get_logdir())
    model_state_dict, optimizer_state_dict, avg_loss = engine(writer=writer, verbose=True, force=True, **constants)
    avg_loss = torch.tensor(avg_loss)
    if writer:
        writer.add_hparams(hyperparams,
                           {'hparam/loss': avg_loss[-20:-1].mean().item()})

    if rank == 0:
        for k in constants:
            if k in ['model', 'train_data', 'val_data', 'train_sampler', 'test_sampler']:
                constants[k] = str(constants[k])

        constants['model_state_dict'] = model_state_dict
        constants['optimizer_state_dict'] = optimizer_state_dict

        torch.save(constants,
                   f'/home/chris/Dropbox (Partners HealthCare)/lsd_pure_torch/models/{os.path.split(writer.log_dir)[-1]}.trch')

    cleanup(rank)
