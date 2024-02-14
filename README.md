# lsd_pure_torch
## Local Shape Descriptors For Neuron Segmentation, but only pytorch. 

This is more or less a rewrite of the work of Arlo Sheridan published here: https://www.nature.com/articles/s41592-022-01711-z
However removing numpy as a dependency. All functions should work with pytorch arrays, be differentiable, and cuda
compatible. 

Additionally, there should be a minimally functional training setup in scr/train/engine.py
This was originally written to be used in the Biomedical Image Segmentation Models (BISM) library: https://github.com/buswinka/bism

To calculate LSD of a 3D instance mask
```python
from lsd_pure_torch import lsd
import torch
import numpy as np
import skimage.io as io
from typing import Tuple

instance_mask: np.ndarray = io.imread('path/to/your/image.tif')  ## shape: [X, Y, Z], dtype: np.uint32
instance_mask: torch.Tensor = torch.from_numpy(instance_mask)
instance_mask = instance_mask.to('cuda').unsqueeze(0) ## shape: [C=1, X, Y, Z], dtype: np.uint32, device='cuda'

sigma: tuple[float, float, float] = (8., 8., 8.)  # Standard deviation affecting shape descriptors. 
voxel_size: tuple[int, int, int] = (1, 1, 5)  ## Relative voxel anisotropy. In this case, Z is 5 time spatially larger than X and Y

# LSD will return a 4D Tensor with shape [C=10, X, Y, Z], dtype=torch.float, device=instance_mask.device
local_shape_descriptors: torch.Tensor = lsd(segmentation=instance_mask, sigma=sigma, voxel_size=voxel_size)  
```

