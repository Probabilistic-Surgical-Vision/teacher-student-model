from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module

from .utils import ImagePyramid

from . import utils as u


class WeightedSSIMError(nn.Module):
    def __init__(self, alpha: float = 0.85, k1: float = 0.01,
                 k2: float = 0.03) -> None:

        super().__init__()

        self.alpha = alpha
        self.k1 = k1 ** 2
        self.k2 = k2 ** 2

        self.pool = nn.AvgPool2d(kernel_size=3, stride=1)

        self.__previous_image_error = None

    @property
    def previous_image_error(self) -> Tensor:
        return self.__previous_image_error

    def ssim(self, x: Tensor, y: Tensor) -> Tensor:

        luminance_x = self.pool(x)
        luminance_y = self.pool(y)

        luminance_xx = luminance_x * luminance_x
        luminance_yy = luminance_y * luminance_y
        luminance_xy = luminance_x * luminance_y

        contrast_x = self.pool(x * x) - luminance_xx
        contrast_y = self.pool(y * y) - luminance_yy

        contrast_xy = self.pool(x * y) - luminance_xy

        numerator = ((2 * luminance_xy) + self.k1) \
            * ((2 * contrast_xy) + self.k2)

        denominator = (luminance_xx + luminance_yy + self.k1) \
            * (contrast_x + contrast_y + self.k2)

        return numerator / denominator

    def dssim(self, x: Tensor, y: Tensor) -> Tensor:
        dissimilarity = (1 - self.ssim(x, y)) / 2
        return torch.clamp(dissimilarity, 0, 1)

    def l1_error(self, x: Tensor, y: Tensor) -> Tensor:
        return (x - y).abs()

    def image_error(self, images: Tensor, recon: Tensor) -> Tensor:
        _, _, height, width = images.size()

        left_l1_error = self.l1_error(images[:, 0:3], recon[:, 0:3])
        right_l1_error = self.l1_error(images[:, 3:6], recon[:, 3:6])

        left_ssim_error = self.dssim(images[:, 0:3], recon[:, 0:3])
        right_ssim_error = self.dssim(images[:, 3:6], recon[:, 3:6])

        l1_error = torch.cat((left_l1_error, right_l1_error), dim=1)
        ssim_error = torch.cat((left_ssim_error, right_ssim_error), dim=1)

        ssim_error = F.interpolate(ssim_error, size=(height, width),
                                   mode='bilinear', align_corners=True)

        return (self.alpha * ssim_error) + ((1 - self.alpha) * l1_error)

    def forward(self, images: Tensor, recon: Tensor) -> Tensor:
        error = self.image_error(images, recon)
        left_error, right_error = torch.split(error, [3, 3], dim=1)

        self.__previous_image_error = error

        return torch.mean(left_error + right_error)


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
                recon_pyramid: ImagePyramid, disc: Module) -> Tensor:

        perceptual_loss = 0

        image_maps = disc.features(image_pyramid)
        recon_maps = disc.features(recon_pyramid)

        for image_map, recon_map in zip(image_maps, recon_maps):
            perceptual_loss += u.l1_loss(image_map, recon_map)

        return perceptual_loss


class AdversarialLoss(nn.Module):
    def __init__(self, loss: str = 'mse') -> None:
        super().__init__()

        self.adversarial = nn.MSELoss() \
            if loss == 'mse' else nn.BCELoss()

    def forward(self, recon_pyramid: ImagePyramid, discriminator: Module,
                is_fake: bool = True) -> Tensor:

        predictions = discriminator(recon_pyramid)
        labels = torch.zeros_like(predictions) \
            if is_fake else torch.ones_like(predictions)

        return self.adversarial(predictions, labels)


class ReprojectionErrorLoss(nn.Module):
    def __init__(self, loss_type: str = 'l1',
                 smoothness_weight: float = 1.0,
                 consistency_weight: float = 1.0,
                 pooling: bool = False) -> None:

        super().__init__()

        if loss_type not in ('l1', 'bayesian', 'log_bayesian'):
            raise ValueError('Loss must be either "l1", "bayesian" '
                             'or "log_bayesian".')

        self.loss_type = loss_type

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

    def bayesian(self, predicted: Tensor, truth: Tensor) -> Tensor:
        return torch.mean((truth / predicted) + torch.log(predicted))

    def log_bayesian(self, predicted: Tensor, truth: Tensor) -> Tensor:
        return torch.mean((truth / torch.exp(-predicted)) + predicted) / 2

    def l1(self, predicted: Tensor, truth: Tensor) -> Tensor:
        return u.l1_loss(predicted, truth)

    def forward(self, predicted: Tensor, truth: Tensor) -> Tensor:
        left, right = torch.split(truth.detach().clone(), [3, 3], dim=1)
        left, right = left.mean(1, keepdim=True), right.mean(1, keepdim=True)

        # We flip left and right since the right reprojection error
        # is given by the left disparity and vice-versa
        truth = torch.cat((right, left), dim=1)

        predicted = self.pool(predicted)
        truth = self.pool(truth)

        loss = self.loss_function(predicted, truth)

        smoothness_loss = self.smoothness(predicted, truth) \
            if self.smoothness_weight > 0 else 0
        consistency_loss = self.consistency(predicted) \
            if self.consistency_weight > 0 else 0

        return loss + (smoothness_loss * self.smoothness_weight) \
            + (consistency_loss * self.consistency_weight)


class ModelLoss(nn.Module):
    def __init__(self, wssim_weight: float = 1.0,
                 consistency_weight: float = 1.0,
                 smoothness_weight: float = 1.0,
                 adversarial_weight: float = 0.85,
                 predictive_error_weight: float = 1.0,
                 perceptual_weight: float = 0.05,
                 wssim_alpha: float = 0.85,
                 perceptual_start: int = 5,
                 adversarial_loss_type: str = 'mse',
                 error_loss_config: Optional[dict] = None) -> None:

        super().__init__()

        self.wssim = WeightedSSIMError(wssim_alpha)

        self.consistency = ConsistencyLoss()
        self.smoothness = SmoothnessLoss()

        self.adversarial = AdversarialLoss(adversarial_loss_type)
        self.perceptual = PerceptualLoss()

        if error_loss_config is None:
            error_loss_config = {}

        self.predictive_error = ReprojectionErrorLoss(**error_loss_config)

        self.perceptual_start = perceptual_start

        self.wssim_weight = wssim_weight
        self.consistency_weight = consistency_weight
        self.smoothness_weight = smoothness_weight
        self.adversarial_weight = adversarial_weight
        self.perceptual_weight = perceptual_weight

        self.predictive_error_weight = predictive_error_weight

        self.__reprojection_errors = []

    @property
    def reprojection_errors(self) -> List[Tensor]:
        return self.__reprojection_errors

    def forward(self, image_pyramid: ImagePyramid,
                disparities: Tuple[Tensor, ...],
                recon_pyramid: ImagePyramid, epoch: Optional[int] = None,
                discriminator: Optional[Module] = None) -> Tensor:

        self.__reprojection_errors = []

        reprojection_loss = 0
        consistency_loss = 0
        smoothness_loss = 0
        adversarial_loss = 0
        perceptual_loss = 0

        error_loss = 0

        scales = zip(image_pyramid, disparities, recon_pyramid)

        for i, (images, disparity, recon_images) in enumerate(scales):
            disparity, uncertainty = torch.split(disparity, [2, 2], dim=1)

            reprojection_loss += self.wssim(images, recon_images)
            consistency_loss += self.consistency(disparity)
            smoothness_loss += self.smoothness(disparity, images) / (2 ** i)

            reprojection_err = self.wssim.previous_image_error
            self.reprojection_errors.append(reprojection_err)

            error_loss += self.predictive_error(uncertainty, reprojection_err)

        if discriminator is not None:
            adversarial_loss += self.adversarial(recon_pyramid, discriminator)

            if epoch is not None and epoch >= self.perceptual_start:
                perceptual_loss += self.perceptual(image_pyramid,
                                                   recon_pyramid,
                                                   discriminator)

        total_disparity_loss = reprojection_loss * self.wssim_weight \
            + (consistency_loss * self.consistency_weight) \
            + (smoothness_loss * self.smoothness_weight) \
            + (adversarial_loss * self.adversarial_weight) \
            + (perceptual_loss * self.perceptual_weight) \
        
        total_error_loss = (error_loss * self.predictive_error_weight)

        return total_disparity_loss, total_error_loss
