import torch
import torch.nn.modules.batchnorm
import skoots.lib.utils
from torch import Tensor
from typing import Dict, Optional, Union, List
import matplotlib.pyplot as plt
from torchvision.utils import flow_to_image, draw_keypoints, make_grid

@torch.no_grad()
def update_bn(loader, model, device=None):
    r"""Updates BatchNorm running_mean, running_var buffers in the model.

    It performs one pass over data in `loader` to estimate the activation
    statistics for BatchNorm layers in the model.
    Args:
        loader (torch.utils.data.DataLoader): dataset loader to compute the
            activation statistics on. Each data batch should be either a
            tensor, or a list/tuple whose first element is a tensor
            containing data.
        model (torch.nn.Module): model for which we seek to update BatchNorm
            statistics.
        device (torch.device, optional): If set, data will be transferred to
            :attr:`device` before being passed into :attr:`model`.

    Example:
        >>> loader, model = ...
        >>> torch.optim.swa_utils.update_bn(loader, model)

    .. note::
        The `update_bn` utility assumes that each data batch in :attr:`loader`
        is either a tensor or a list or tuple of tensors; in the latter case it
        is assumed that :meth:`model.forward()` should be called on the first
        element of the list or tuple corresponding to the data batch.
    """
    momenta = {}
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum

    if not momenta:
        return

    was_training = model.training
    model.train()
    for module in momenta.keys():
        module.momentum = None
        module.num_batches_tracked *= 0

    for batch_ind, data_dict in enumerate(loader):
        images, data_dict = hcat.lib.utils.prep_dict(data_dict, device)  # Preps output to handle potential batched data
        output = model(images, data_dict)

    # for input in loader:
    #     if isinstance(input, (list, tuple)):
    #         input = input[0]
    #     if device is not None:
    #         input = input.to(device)
    #
    #     model(input)

    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]
    model.train(was_training)


@torch.jit.script
def sum_loss(input: Dict[str, Tensor]) -> Optional[Tensor]:
    loss: Union[None, Tensor] = None
    for key in input:
        loss = loss + input[key] if loss is not None else input[key]
    return loss


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def mask_overlay(image: Tensor, mask: Tensor, thr: Optional[float] = 0.5) -> Tensor:
    if image.ndim > 3:
        raise RuntimeError('3D Tensors not supported!!!', image.shape)

    _image = image.gt(thr)
    _mask = mask.gt(thr)

    _, x, y = _image.shape

    overlap = _image * _mask
    false_positive = torch.logical_not(_image) * mask
    false_negative = torch.logical_not(_mask) * image

    out = torch.zeros((3, x, y), device=image.device)

    out[:, overlap[0, ...].gt(0.5)] = 1.
    out[0, false_positive[0, ...].gt(0.5)] = 0.5
    out[2, false_negative[0, ...].gt(0.5)] = 0.5

    return out


def write_progress(writer, tag, epoch, images, masks, lsd, out):

    _a = images[0, [0, 0, 0], :, :, 7].cpu()
    _b = masks[0, [0, 0, 0], :, :, 7].gt(0.5).float().cpu()

    img_list = [_a, _b]

    for index in [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 9, 9]]:
        img_list.append(lsd[0, index, ..., 7].float().cpu())
        img_list.append(out[0, index, ..., 7].float().cpu())

    _img = make_grid(img_list, nrow=1, normalize=True, scale_each=True)

    writer.add_image(tag, _img, epoch, dataformats='CWH')
