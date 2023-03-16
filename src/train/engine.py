from typing import List, Tuple, Callable, Union, OrderedDict, Optional
import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models.detection import FasterRCNN

from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from tqdm import trange
from torch.cuda.amp import GradScaler, autocast
from statistics import mean
from torchvision.utils import flow_to_image, draw_keypoints, make_grid
import matplotlib.pyplot as plt
import torch.optim.swa_utils

from src.train.utils import write_progress
Dataset = Union[Dataset, DataLoader]

# SKELETON TRAINING ENGINE
def engine(
        model,
        lr: float,
        wd: float,
        epochs: int,
        optimizer: Optimizer,
        scheduler,
        loss_fn,
        device: str,
        savepath: str,
        train_data: Dataset,
        rank: int,
        val_data: Optional[Dataset] = None,
        train_sampler=None,
        test_sampler=None,
        writer=None,
        verbose=False,
        distributed=True,
        mixed_precision=False,
        n_warmup: int = 100,
        force=False,
        **kwargs,
) -> Tuple[OrderedDict, OrderedDict, List[float]]:

    # Print out each kwarg to std out
    if verbose and rank == 0:
        print('Initiating Training Run', flush=False)
        vars = locals()
        for k in vars:
            if k != 'model':
                print(f'\t> {k}: {vars[k]}', flush=False)
        print('', flush=True)


    optimizer = optimizer(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = scheduler(optimizer)
    scaler = GradScaler(enabled=mixed_precision)

    swa_model = torch.optim.swa_utils.AveragedModel(model)
    swa_start = 100
    swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, swa_lr=0.05)

    # Save each loss value in a list...
    avg_epoch_loss = []
    avg_epoch_embed_loss = []
    avg_epoch_prob_loss = []
    avg_epoch_skele_loss = []

    avg_val_loss = []
    avg_val_embed_loss = []
    avg_val_prob_loss = []
    avg_val_skele_loss = []

    # skel_crossover_loss = skoots.train.loss.split(n_iter=3, alpha=2)

    # Warmup... Get the first from train_data
    for images, masks, lsd in train_data:
        pass

    assert images is not None, len(train_data)

    warmup_range = trange(n_warmup, desc='Warmup: {}')
    for w in warmup_range:
        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=mixed_precision):  # Saves Memory!
            out: Tensor = model(images)
            loss = loss_fn(out, lsd)

            warmup_range.desc = f'{loss.item()}'

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    # Train Step...
    epoch_range = trange(epochs, desc=f'Loss = {1.0000000}') if rank == 0 else range(epochs)
    for e in epoch_range:
        _loss = []

        if distributed:
            train_sampler.set_epoch(e)

        for images, masks, lsd in train_data:
            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=mixed_precision):  # Saves Memory!
                out: Tensor = model(images)
                loss = loss_fn(out, lsd)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if e > swa_start:
                swa_model.update_parameters(model)

            _loss.append(loss.item())

        avg_epoch_loss.append(mean(_loss))
        scheduler.step()

        if writer and (rank == 0):
            writer.add_scalar('lr', scheduler.get_last_lr()[-1], e)
            writer.add_scalar('Loss/train', avg_epoch_loss[-1], e)
            write_progress(writer=writer, tag='Train', epoch=e, images=images, masks=masks,
                           lsd=lsd, out=out)

        # # Validation Step
        if e % 10 == 0 and val_data:
            _loss = []
            for images, masks, lsd in val_data:
                with autocast(enabled=mixed_precision):  # Saves Memory!
                    with torch.no_grad():
                        out: Tensor = swa_model(images)
                        loss = loss_fn(out, lsd)


                scaler.scale(loss)
                _loss.append(loss.item())

            avg_val_loss.append(mean(_loss))

            if writer and (rank == 0):
                writer.add_scalar('Validation/train', avg_val_loss[-1], e)
                write_progress(writer=writer, tag='Train', epoch=e, images=images, masks=masks,
                               lsd=lsd, out=out)

        if rank == 0:
            epoch_range.desc = f'lr={scheduler.get_last_lr()[-1]:.3e}, Loss (train | val): ' + f'{avg_epoch_loss[-1]:.5f} | {avg_val_loss[-1]:.5f}'

        state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        if e % 100 == 0:
            torch.save(state_dict, savepath + f'/test_{e}.trch')

    return state_dict, optimizer.state_dict(), avg_val_loss
