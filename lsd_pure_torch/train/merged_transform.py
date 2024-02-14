import torch
from torch import Tensor
import torchvision.transforms.functional as ttf
from typing import Dict, Tuple, Union, Sequence, List, Callable, Optional
from lsd_pure_torch.lib.morphology import binary_erosion, _get_binary_kernel3d
from lsd_pure_torch.lib.lsd import lsd

import math
from copy import deepcopy


@torch.jit.script
def _get_box(mask: Tensor, device: str, threshold: int) -> Tuple[Tensor, Tensor]:
    # mask in shape of 300, 400, 1 [H, W, z=1]
    nonzero = torch.nonzero(mask)  # Q, 3=[x,y,z]
    label = mask.max()

    box = torch.tensor([-1, -1, -1, -1], dtype=torch.long, device=device)

    # Recall, image in shape of [C, H, W]

    if nonzero.numel() > threshold:
        x0 = torch.min(nonzero[:, 1])
        x1 = torch.max(nonzero[:, 1])
        y0 = torch.min(nonzero[:, 0])
        y1 = torch.max(nonzero[:, 0])

        if (x1 - x0 > 0) and (y1 - y0 > 0):
            box[0] = x0
            box[1] = y0
            box[2] = x1
            box[3] = y1

    return label, box


@torch.jit.script
def _get_affine_matrix(
        center: List[float], angle: float, translate: List[float], scale: float, shear: List[float], device: str,
) -> Tensor:
    # We need compute the affine transformation matrix: M = T * C * RSS * C^-1

    rot = math.radians(angle)
    sx, sy = [math.radians(s) for s in shear]

    T: Tensor = torch.eye(3, device=device)
    T[0, -1] = translate[0]
    T[1, -1] = translate[1]

    C: Tensor = torch.eye(3, device=device)
    C[0, -1] = center[0]
    C[1, -1] = center[1]

    # RSS without scaling
    a = math.cos(rot - sy) / math.cos(sy)
    b = -math.cos(rot - sy) * math.tan(sx) / math.cos(sy) - math.sin(rot)
    c = math.sin(rot - sy) / math.cos(sy)
    d = -math.sin(rot - sy) * math.tan(sx) / math.cos(sy) + math.cos(rot)

    RSS = torch.tensor(
        [[a, b, 0.],
         [c, d, 0.],
         [0., 0., 1.]],
        device=device)
    RSS = RSS * scale
    RSS[-1, -1] = 1

    return T @ C @ RSS @ torch.inverse(C)

@torch.no_grad()
def merged_transform_3D(data_dict: Dict[str, Tensor],
                        device: Optional[str] = None,
                        ) -> Dict[str, Tensor]:
    DEVICE: str = str(data_dict['image'].device) if device is None else device

    # Image should be in shape of [C, H, W, D]
    CROP_WIDTH = torch.tensor([300], device=DEVICE)
    CROP_HEIGHT = torch.tensor([300], device=DEVICE)
    CROP_DEPTH = torch.tensor([20], device=DEVICE)

    FLIP_RATE = torch.tensor(0.5, device=DEVICE)

    BRIGHTNESS_RATE = torch.tensor(0.4, device=DEVICE)
    BRIGHTNESS_RANGE = torch.tensor((-0.1, 0.1), device=DEVICE)

    NOISE_GAMMA = torch.tensor(0.1, device=DEVICE)
    NOISE_RATE = torch.tensor(0.2, device=DEVICE)

    FILTER_RATE = torch.tensor(0.5, device=DEVICE)

    CONTRAST_RATE = torch.tensor(0.33, device=DEVICE)
    CONTRAST_RANGE = torch.tensor((0.75, 2.), device=DEVICE)

    AFFINE_RATE = torch.tensor(0.66, device=DEVICE)
    AFFINE_SCALE = torch.tensor((0.85, 1.1), device=DEVICE)
    AFFINE_YAW = torch.tensor((-180, 180), device=DEVICE)
    AFFINE_SHEAR = torch.tensor((-7, 7), device=DEVICE)

    masks = torch.clone(data_dict['masks'])  # .to(DEVICE))
    image = torch.clone(data_dict['image'])  #


    # ------------ Random Crop 1
    extra = 0#300
    w = CROP_WIDTH + extra if CROP_WIDTH + extra <= image.shape[1] else torch.tensor(image.shape[1])
    h = CROP_HEIGHT + extra if CROP_HEIGHT + extra <= image.shape[2] else torch.tensor(image.shape[2])
    d = CROP_DEPTH if CROP_DEPTH <= image.shape[3] else torch.tensor(image.shape[3])

    # select a random point for croping
    x0 = torch.randint(0, image.shape[1] - w.item() + 1, (1,), device=DEVICE)
    y0 = torch.randint(0, image.shape[2] - h.item() + 1, (1,), device=DEVICE)
    z0 = torch.randint(0, image.shape[3] - d.item() + 1, (1,), device=DEVICE)

    x1 = x0 + w
    y1 = y0 + h
    z1 = z0 + d

    image = image[:, x0.item():x1.item(), y0.item():y1.item(), z0.item():z1.item()].to(DEVICE)
    masks = masks[:, x0.item():x1.item(), y0.item():y1.item(), z0.item():z1.item()].to(DEVICE)


    # # -------------------affine (Cant use baked skeletons)
    # if torch.rand(1, device=DEVICE) < AFFINE_RATE:
    #     angle = (AFFINE_YAW[1] - AFFINE_YAW[0]) * torch.rand(1, device=DEVICE) + AFFINE_YAW[0]
    #     shear = (AFFINE_SHEAR[1] - AFFINE_SHEAR[0]) * torch.rand(1, device=DEVICE) + AFFINE_SHEAR[0]
    #     scale = (AFFINE_SCALE[1] - AFFINE_SCALE[0]) * torch.rand(1, device=DEVICE) + AFFINE_SCALE[0]
    #
    #
    #     image = ttf.affine(image.permute(0, 3, 1, 2).float(),
    #                        angle=angle.item(),
    #                        shear=[float(shear.item())],
    #                        scale=scale.item(),
    #                        translate=[0, 0]).permute(0, 2, 3, 1)
    #
    #     masks = ttf.affine(masks.permute(0, 3, 1, 2).float(),
    #                        angle=angle.item(),
    #                        shear=[float(shear.item())],
    #                        scale=scale.item(),
    #                        translate=[0, 0]).permute(0, 2, 3, 1)
    #
    #
    #
    # # ------------ Center Crop 2
    # w = CROP_WIDTH if CROP_WIDTH < image.shape[1] else torch.tensor(image.shape[1])
    # h = CROP_HEIGHT if CROP_HEIGHT < image.shape[2] else torch.tensor(image.shape[2])
    # d = CROP_DEPTH if CROP_DEPTH < image.shape[3] else torch.tensor(image.shape[3])
    #
    # # Center that instance
    # x0 = torch.randint(0, image.shape[1] - w.item() + 1, (1,), device=DEVICE)
    # y0 = torch.randint(0, image.shape[2] - h.item() + 1, (1,), device=DEVICE)
    # z0 = torch.randint(0, image.shape[3] - d.item() + 1, (1,), device=DEVICE)
    #
    # x1 = x0 + w
    # y1 = y0 + h
    # z1 = z0 + d
    #
    # image = image[:, x0.item():x1.item(), y0.item():y1.item(), z0.item():z1.item()].to(DEVICE)
    # masks = masks[:, x0.item():x1.item(), y0.item():y1.item(), z0.item():z1.item()].to(DEVICE)
    #
    #
    # # ------------------- x flip
    # if torch.rand(1, device=DEVICE) < FLIP_RATE:
    #     image = image.flip(1)
    #     masks = masks.flip(1)
    #
    # # ------------------- y flip
    # if torch.rand(1, device=DEVICE) < FLIP_RATE:
    #     image = image.flip(2)
    #     masks = masks.flip(2)
    #
    # # ------------------- z flip
    # if torch.rand(1, device=DEVICE) < FLIP_RATE:
    #     image = image.flip(3)
    #     masks = masks.flip(3)
    #
    #
    # # # ------------------- Random Invert
    # if torch.rand(1, device=DEVICE) < BRIGHTNESS_RATE:
    #     image = image.sub(1).mul(-1)
    #
    # # ------------------- Adjust Brightness
    # if torch.rand(1, device=DEVICE) < BRIGHTNESS_RATE:
    #     # funky looking but FAST
    #     val = torch.empty(image.shape[0], device=DEVICE).uniform_(BRIGHTNESS_RANGE[0], BRIGHTNESS_RANGE[1])
    #     image = image.add(val.reshape(image.shape[0], 1, 1, 1)).clamp(0, 1)
    #
    # # ------------------- Adjust Contrast
    # if torch.rand(1, device=DEVICE) < CONTRAST_RATE:
    #     contrast_val = (CONTRAST_RANGE[1] - CONTRAST_RANGE[0]) * torch.rand((image.shape[0]), device=DEVICE) + \
    #                    CONTRAST_RANGE[0]
    #
    #     for z in range(image.shape[-1]):
    #         image[..., z] = ttf.adjust_contrast(image[..., z], contrast_val[0].item()).squeeze(0)
    #
    # # ------------------- Noise
    # if torch.rand(1, device=DEVICE) < NOISE_RATE:
    #     noise = torch.rand(image.shape, device=DEVICE) * NOISE_GAMMA
    #
    #     image = image.add(noise).clamp(0, 1)

    data_dict['image'] = image
    data_dict['masks'] = masks
    data_dict['lsd'] = lsd(masks, sigma=(8, 8, 8), voxel_size=(1, 1, 5))

    return data_dict

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import torchvision.utils
    import tqdm
    import torch.distributed as dist
    import torch.optim.lr_scheduler
    from lsd_pure_torch.train.dataloader import dataset, colate, MultiDataset

    from torch.utils.data import DataLoader

    torch.manual_seed(0)

    path = '/home/chris/Dropbox (Partners HealthCare)/trainMitochondriaSegmentation/data/unscaled/train'
    data = dataset(path=path, transforms=merged_transform_3D, sample_per_image=4).to('cuda')
    dataloader = DataLoader(data, num_workers=0, batch_size=4, collate_fn=colate)

    for im, ma, lsd in dataloader:
        print(f'{im.shape=}, {ma.shape=}, {lsd.shape=}')
        pass
