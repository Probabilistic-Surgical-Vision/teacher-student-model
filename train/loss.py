from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module

from .utils import ImagePyramid

from . import utils as u


class ConsistencyLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, disp: Tensor) -> Tensor:
        left_lr_disp = u.reconstruct_left_image(disp[:, 0:1], disp[:, 1:2])
        right_lr_disp = u.reconstruct_right_image(disp[:, 1:2], disp[:, 0:1])

        left_con_loss = u.l1_loss(disp[:, 0:1], left_lr_disp)
        right_con_loss = u.l1_loss(disp[:, 1:2], right_lr_disp)

        return torch.sum(left_con_loss + right_con_loss)


class SmoothnessLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def gradient_x(self, x: Tensor) -> Tensor:
        # Pad input to keep output size consistent
        x = F.pad(x, (0, 1, 0, 0), mode='replicate')
        return x[:, :, :, :-1] - x[:, :, :, 1:]

    def gradient_y(self, x: Tensor) -> Tensor:
        # Pad input to keep output size consistent
        x = F.pad(x, (0, 0, 0, 1), mode='replicate')
        return x[:, :, :-1, :] - x[:, :, 1:, :]

    def smoothness_weights(self, image_gradient: Tensor) -> Tensor:
        return torch.exp(-image_gradient.abs().mean(dim=1, keepdim=True))

    def loss(self, disparity: Tensor, image: Tensor) -> Tensor:
        disp_grad_x = self.gradient_x(disparity)
        disp_grad_y = self.gradient_y(disparity)

        image_grad_x = self.gradient_x(image)
        image_grad_y = self.gradient_y(image)

        weights_x = self.smoothness_weights(image_grad_x)
        weights_y = self.smoothness_weights(image_grad_y)

        smoothness_x = disp_grad_x * weights_x
        smoothness_y = disp_grad_y * weights_y

        return smoothness_x.abs() + smoothness_y.abs()

    def forward(self, disp: Tensor, images: Tensor) -> Tensor:
        smooth_left_loss = self.loss(disp[:, 0:1], images[:, 0:3])
        smooth_right_loss = self.loss(disp[:, 1:2], images[:, 3:6])

        return torch.mean(smooth_left_loss + smooth_right_loss)


class PerceptualLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, image_pyramid: ImagePyramid,
                truth_pyramid: ImagePyramid, disc: Module) -> Tensor:

        perceptual_loss = 0

        image_maps = disc.features(image_pyramid)
        recon_maps = disc.features(truth_pyramid)

        for image_map, recon_map in zip(image_maps, recon_maps):
            perceptual_loss += u.l1_loss(image_map, recon_map)

        return perceptual_loss


class AdversarialLoss(nn.Module):
    def __init__(self, loss: str = 'mse') -> None:
        super().__init__()

        self.adversarial = nn.MSELoss() \
            if loss == 'mse' else nn.BCELoss()

    def forward(self, truth_pyramid: ImagePyramid, discriminator: Module,
                is_fake: bool = True) -> Tensor:

        predictions = discriminator(truth_pyramid)
        labels = torch.zeros_like(predictions) \
            if is_fake else torch.ones_like(predictions)

        return self.adversarial(predictions, labels)


class DisparityErrorLoss(nn.Module):
    def __init__(self, loss_type: str = 'l1',
                 smoothness_weight: float = 1.0,
                 consistency_weight: float = 1.0,
                 include_aleatoric: bool = False,
                 pooling: bool = False) -> None:

        super().__init__()

        if loss_type not in ('l1', 'bayesian', 'log_bayesian'):
            raise ValueError('Loss must be either "l1", "bayesian" '
                             'or "log_bayesian".')

        self.loss_type = loss_type
        self.include_aleatoric = include_aleatoric

        if loss_type == 'l1':
            self.loss_function = self.l1
        elif loss_type == 'bayesian':
            self.loss_function = self.bayesian
        else:
            self.loss_function = self.log_bayesian

        self.smoothness_weight = smoothness_weight
        self.consistency_weight = consistency_weight

        self.smoothness = SmoothnessLoss() \
            if smoothness_weight > 0 else None

        self.consistency = ConsistencyLoss() \
            if consistency_weight > 0 else None
        
        self.pool = nn.AvgPool2d(kernel_size=3, stride=1) \
            if pooling else nn.Identity()

        self.__error_map = None

    @property
    def error_map(self) -> Tensor:
        return self.__error_map

    def bayesian(self, predicted: Tensor, truth: Tensor) -> Tensor:
        return torch.mean((truth / predicted) + torch.log(predicted))

    def log_bayesian(self, predicted: Tensor, truth: Tensor) -> Tensor:
        return torch.mean((truth / torch.exp(-predicted)) + predicted) / 2

    def l1(self, predicted: Tensor, truth: Tensor) -> Tensor:
        return u.l1_loss(predicted, truth)

    def forward(self, predicted: Tensor, truth: Tensor) -> Tensor:
        pred_disp, pred_error = torch.split(predicted, [2, 2], dim=1)
        true_disp, true_error = torch.split(truth, [2, 2], dim=1)

        if self.include_aleatoric:
            aleatoric = (pred_disp.detach().clone() - true_disp).abs()
            true_error = torch.sqrt((true_error ** 2) + (aleatoric ** 2))

        self.__error_map = true_error

        pred_error = self.pool(pred_error)
        true_error = self.pool(true_error)

        loss = self.loss_function(pred_error, true_error)

        smoothness_loss = self.smoothness(pred_error, true_error) \
            if self.smoothness_weight > 0 else 0
        consistency_loss = self.consistency(pred_error) \
            if self.consistency_weight > 0 else 0

        return loss + (smoothness_loss * self.smoothness_weight) \
            + (consistency_loss * self.consistency_weight)


class TukraEnsembleLoss(nn.Module):
    def __init__(self, disparity_weight: float = 1.0,
                 consistency_weight: float = 1.0,
                 smoothness_weight: float = 1.0,
                 adversarial_weight: float = 0.85,
                 predictive_error_weight: float = 1.0,
                 perceptual_weight: float = 0.05,
                 perceptual_start: int = 5,
                 adversarial_loss_type: str = 'mse',
                 error_loss_config: Optional[dict] = None) -> None:

        super().__init__()

        self.consistency = ConsistencyLoss()
        self.smoothness = SmoothnessLoss()

        self.adversarial = AdversarialLoss(adversarial_loss_type)
        self.perceptual = PerceptualLoss()

        if error_loss_config is None:
            error_loss_config = {}

        self.predictive_error = DisparityErrorLoss(**error_loss_config)

        self.perceptual_start = perceptual_start

        self.disparity_weight = disparity_weight
        self.consistency_weight = consistency_weight
        self.smoothness_weight = smoothness_weight
        self.adversarial_weight = adversarial_weight
        self.perceptual_weight = perceptual_weight

        self.predictive_error_weight = predictive_error_weight

        self.__error_maps = []

    @property
    def error_maps(self) -> List[Tensor]:
        return self.__error_maps

    def forward(self, image_pyramid: ImagePyramid,
                disparities: Tuple[Tensor, ...],
                truth_pyramid: ImagePyramid, epoch: Optional[int] = None,
                discriminator: Optional[Module] = None) -> Tensor:

        self.__error_maps = []

        disparity_loss = 0
        consistency_loss = 0
        smoothness_loss = 0
        adversarial_loss = 0
        perceptual_loss = 0

        error_loss = 0

        scales = zip(image_pyramid, disparities, truth_pyramid)

        for i, (images, prediction, truth) in enumerate(scales):
            pred_disp, true_disp = prediction[:, :2], truth[:, :2]

            disparity_loss += F.l1_loss(pred_disp, true_disp)
            consistency_loss += self.consistency(pred_disp)
            smoothness_loss += self.smoothness(pred_disp, images) / (2 ** i)

            error_loss += self.predictive_error(prediction, truth)
            error_map = self.predictive_error.error_map

            self.error_maps.append(error_map)

        if discriminator is not None:
            adversarial_loss += self.adversarial(truth_pyramid, discriminator)

            if epoch is not None and epoch >= self.perceptual_start:
                perceptual_loss += self.perceptual(image_pyramid,
                                                   truth_pyramid,
                                                   discriminator)

        total_disparity_loss = disparity_loss * self.disparity_weight \
            + (consistency_loss * self.consistency_weight) \
            + (smoothness_loss * self.smoothness_weight) \
            + (adversarial_loss * self.adversarial_weight) \
            + (perceptual_loss * self.perceptual_weight) \
        
        total_error_loss = (error_loss * self.predictive_error_weight)

        return total_disparity_loss, total_error_loss
