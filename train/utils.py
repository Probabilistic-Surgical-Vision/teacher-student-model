from typing import List, Optional, OrderedDict, Union

import matplotlib.pyplot as plt

import numpy as np

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module

from torchvision.utils import make_grid

ImagePyramid = List[Tensor]
Device = Union[torch.device, str]


def l1_loss(x: Tensor, y: Tensor) -> Tensor:
    return (x - y).abs().mean()


def scale_pyramid(x: Tensor, scales: int) -> ImagePyramid:
    _, _, height, width = x.size()

    pyramid = []

    for i in range(scales):
        ratio = 2 ** i

        size = (height // ratio, width // ratio)
        x_resized = F.interpolate(x, size=size, mode='bilinear',
                                  align_corners=True)

        pyramid.append(x_resized)

    return pyramid


def detach_pyramid(pyramid: ImagePyramid) -> ImagePyramid:
    return [layer.detach().clone() for layer in pyramid]


def reconstruct(disparity: Tensor, opposite_image: Tensor) -> Tensor:
    batch_size, _, height, width = opposite_image.size()

    # Original coordinates of pixels
    x_base = torch.linspace(0, 1, width) \
        .repeat(batch_size, height, 1) \
        .type_as(opposite_image)

    y_base = torch.linspace(0, 1, height) \
        .repeat(batch_size, width, 1) \
        .transpose(1, 2) \
        .type_as(opposite_image)

    # Apply shift in X direction
    x_shifts = disparity.squeeze(dim=1)

    # In grid_sample coordinates are assumed to be between -1 and 1
    flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)
    flow_field = (2 * flow_field) - 1

    return F.grid_sample(opposite_image, flow_field, mode='bilinear',
                         padding_mode='zeros')


def reconstruct_left_image(left_disparity: Tensor,
                           right_image: Tensor) -> Tensor:

    return reconstruct(-left_disparity, right_image)


def reconstruct_right_image(right_disparity: Tensor,
                            left_image: Tensor) -> Tensor:

    return reconstruct(right_disparity, left_image)


def reconstruct_pyramid(disparities: ImagePyramid,
                        pyramid: ImagePyramid) -> ImagePyramid:

    recon_pyramid = []

    for disparity, images in zip(disparities, pyramid):
        left_disp, right_disp = torch.split(disparity[:, :2], [1, 1], 1)
        left_image, right_image = torch.split(images, [3, 3], 1)

        left_recon = reconstruct_left_image(left_disp, right_image)
        right_recon = reconstruct_right_image(right_disp, left_image)

        recon_image = torch.cat([left_recon, right_recon], dim=1)
        recon_pyramid.append(recon_image)

    return recon_pyramid


def concatenate_pyramids(a: ImagePyramid, b: ImagePyramid) -> ImagePyramid:
    return [torch.cat((x, y), 0) for x, y in zip(a, b)]


def adjust_disparity_scale(epoch: int, m: float = 0.02, c: float = 0.0,
                           step: float = 0.2, offset: float = 0.1,
                           min_scale: float = 0.3,
                           max_scale: float = 1.0) -> float:

    # Transform epoch to continuous scale using m and c
    scale = (epoch * m) + c
    # Quantise to fit the grid defined by step and offset
    scale = (round((scale + offset) / step) * step) - offset
    # Clip to between min and max bounds
    return np.clip(scale, min_scale, max_scale)


def to_heatmap(x: Tensor, device: Device = 'cpu', inverse: bool = False,
               colour_map: str = 'inferno') -> Tensor:

    image = x.squeeze(0).cpu().numpy()
    image = 1 - image if inverse else image

    transform = plt.get_cmap(colour_map)
    heatmap = transform(image)[:, :, :3]  # remove alpha channel

    return torch.from_numpy(heatmap).to(device).permute(2, 0, 1)


def combine_disparity(left: Tensor, right: Tensor, device: Device = 'cpu',
                      alpha: float = 20, beta: float = 0.05) -> Tensor:

    left_disp = left.cpu().numpy()
    right_disp = right.cpu().numpy()
    mean_disp = (left_disp + right_disp) / 2

    _, height, width = mean_disp.shape

    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    xv, _ = np.meshgrid(x, y)

    left_mask = 1 - np.clip(alpha * (xv - beta), 0, 1)
    right_mask = np.fliplr(left_mask)

    mean_mask = 1 - (left_mask + right_mask)

    combined_disparity = (right_mask * left_disp) \
        + (left_mask * right_disp) \
        + (mean_mask * mean_disp)

    return torch.from_numpy(combined_disparity).to(device)


def run_discriminator(image_pyramid: ImagePyramid,
                      recon_pyramid: ImagePyramid,
                      discriminator: Module,
                      disc_loss_function: Module,
                      batch_size: int) -> Tensor:

    recon_pyramid = detach_pyramid(recon_pyramid)
    pyramid = concatenate_pyramids(image_pyramid, recon_pyramid)

    predictions = discriminator(pyramid)

    labels = torch.zeros_like(predictions)
    labels[:batch_size] = 1

    return disc_loss_function(predictions, labels) / 2


def get_comparison(image: Tensor, prediction: Tensor, extra: Optional[Tensor],
                   add_scaled: bool = False, device: Device = 'cpu') -> Tensor:

    left_image, right_image = torch.split(image, [3, 3], dim=0)
    left_pred, right_pred = torch.split(prediction, [1, 1], dim=0)

    min_pred, max_pred = prediction.min(), prediction.max()
    scaled_left_pred = (left_pred - min_pred) / (max_pred - min_pred)
    scaled_right_pred = (right_pred - min_pred) / (max_pred - min_pred)

    left_pred = to_heatmap(left_pred, device)
    right_pred = to_heatmap(right_pred, device)

    if extra is not None:
        extra_split = [3, 3] if extra.size(0) == 6 else [1, 1]
        left_extra, right_extra = torch.split(extra, extra_split, dim=0)

        if extra.size(0) == 2:
            left_extra = to_heatmap(left_extra, device)
            right_extra = to_heatmap(right_extra, device)

    images = torch.stack((left_image, right_image, left_pred, right_pred))

    if add_scaled:
        scaled_left_pred = to_heatmap(scaled_left_pred, device)
        scaled_right_pred = to_heatmap(scaled_right_pred, device)

        images = torch.cat((images, scaled_left_pred.unsqueeze(0),
                           scaled_right_pred.unsqueeze(0)))

    if extra is not None:
        images = torch.cat((images, left_extra.unsqueeze(0),
                           right_extra.unsqueeze(0)))

    return make_grid(images, nrow=2)


def prepare_state_dict(state_dict: OrderedDict) -> dict:
    return {k.replace("module.", ""): v for k, v in state_dict.items()}
