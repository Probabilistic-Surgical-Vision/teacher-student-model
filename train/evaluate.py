import os.path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader

from torchvision.utils import save_image

from torchmetrics.functional import \
    structural_similarity_index_measure as ssim

import tqdm

from .loss import WeightedSSIMLoss

from . import sparsification as spars

from . import utils as u
from .utils import Device


def save_comparisons(image: Tensor, disparity: Tensor, uncertainty: Tensor,
                     recon: Tensor, error: Tensor, directory: str,
                     epoch_number: Optional[int] = None,
                     is_final: bool = True, device: Device = 'cpu') -> None:
    """Save a .png comparing the original image, prediction and error images.

    Args:
        image (Tensor): The original image pair.
        prediction (Tensor): The disparity image pair.
        ensemble (Tensor): The ensemblestructed image pair.
        error (Tensor): The true error image pair.
        directory (str): The directory to save the result to.
        epoch_number (Optional[int], optional): The epoch number for creating
            a filename. Defaults to None.
        is_final (bool, optional): Forces the filename to be `final.png`.
            Defaults to True.
        device (Device, optional): The torch device to use. Defaults to 'cpu'.
    """
    prediction_image = u.get_comparison(image, disparity, uncertainty,
                                        scale=False, device=device)
    disparity_image = u.get_comparison(image, disparity, recon,
                                       scale=True, device=device)
    uncertainty_image = u.get_comparison(image, uncertainty, error,
                                         scale=True, device=device)

    dirname = 'final' if is_final else f'epoch_{epoch_number:03}'
    epoch_directory = os.path.join(directory, dirname)

    if not os.path.isdir(epoch_directory):
        os.makedirs(epoch_directory, exist_ok=True)

    print(f'Saving comparisons to:\n\t{epoch_directory}')
    prediction_filename = os.path.join(epoch_directory, 'prediction.png')
    disparity_filename = os.path.join(epoch_directory, 'disparity.png')
    uncertainty_filename = os.path.join(epoch_directory, 'uncertainty.png')

    save_image(prediction_image, prediction_filename)
    save_image(disparity_image, disparity_filename)
    save_image(uncertainty_image, uncertainty_filename)


@torch.no_grad()
def evaluate_model(model: Module, loader: DataLoader,
                   save_evaluation_to: Optional[str] = None,
                   epoch_number: Optional[int] = None,
                   scale: int = 4, is_final: bool = True,
                   kernel_size: int = 11,
                   no_pbar: bool = False,
                   device: Device = 'cpu',
                   rank: int = 0) -> Tuple[float, float]:
    """Loop through the validation set and report model losses.

    Args:
        model (Module): The model to test.
        loader (DataLoader): The validation loader to iterate through.
        loss_function (Module): The loss function to report.
        scale (float, optional): A multiplier to scale the model's predictions
            by. Defaults to 1.0.
        disc (Optional[Module], optional): The discriminator to test.
            Defaults to None.
        disc_loss_function (Optional[Module], optional): The discriminator
            loss functions to report. Defaults to None.
        save_evaluation_to (Optional[str], optional): Path to the directory to
            save comparison images to. Defaults to None.
        epoch_number (Optional[int], optional): The epoch number in training.
            Defaults to None.
        is_final (bool, optional): The evaluation is taking place
            post-training. Defaults to True.
        scales (int, optional): The size of the image pyramid. Defaults to 4.
        device (Device, optional): The torch device to use. Defaults to 'cpu'.
        no_pbar (bool, optional): Disable `tqdm` progress bar. Defaults to
            False.
        rank (int, optional): The rank of the process (for multiprocessing
            only). Defaults to 0.

    Returns:
        float: The average disparity loss per image.
        float: The average uncertainty loss per image.
        float: The average discriminator loss per image.
    """
    running_left_ssim = 0
    running_right_ssim = 0
    running_ause = 0
    running_aurg = 0

    average_left_ssim = None
    average_right_ssim = None
    average_ause = None
    average_aurg = None

    batch_size = loader.batch_size \
        if loader.batch_size is not None \
        else len(loader)

    description = 'Evaluation'
    tepoch = tqdm.tqdm(loader, description, unit='batch',
                       disable=(no_pbar or rank > 0))

    # Set alpha to one so L1 has zero weight
    ssim_loss = WeightedSSIMLoss(alpha=1)

    model.eval()

    for i, image_pair in enumerate(tepoch):
        left = image_pair['left'].to(device)
        right = image_pair['right'].to(device)

        images = torch.cat([left, right], dim=1)

        prediction = model(left, scale)

        disparity, uncertainty = torch.split(prediction, [2, 2], dim=1)
        left_disp, right_disp = torch.split(disparity, [1, 1], dim=1)

        left_recon = u.reconstruct_left_image(left_disp, right)
        right_recon = u.reconstruct_right_image(right_disp, left)

        left_ssim = ssim(left_recon, left, kernel_size=kernel_size,
                         reduction='sum', data_range=1.0)

        right_ssim = ssim(right_recon, right, kernel_size=kernel_size,
                          reduction='sum', data_range=1.0)

        recon = torch.cat((left_recon, right_recon), dim=1)
        _, _, height, width = recon.shape

        error = ssim_loss.image_error(images, recon)
        error = F.interpolate(error, size=(height, width), mode='bilinear',
                              align_corners=True)

        oracle_spars = spars.curve(error, error, device=device)
        pred_spars = spars.curve(error, uncertainty, device=device)
        random_spars = spars.random_curve(error)

        ause = spars.ause(oracle_spars, pred_spars)
        aurg = spars.aurg(pred_spars, random_spars)

        if rank > 0:
            continue

        running_left_ssim += left_ssim.item()
        average_left_ssim = running_left_ssim / ((i+1) * batch_size)

        running_right_ssim += right_ssim.item()
        average_right_ssim = running_right_ssim / ((i+1) * batch_size)

        running_ause += ause.item()
        average_ause = running_ause / ((i+1) * batch_size)

        running_aurg += aurg.item()
        average_aurg = running_aurg / ((i+1) * batch_size)

        tepoch.set_postfix(left=average_left_ssim, right=average_right_ssim,
                           ause=average_ause, aurg=average_aurg, scale=scale)

        if save_evaluation_to is not None and i == 0:
            save_comparisons(images[0], disparity[0], uncertainty[0],
                             recon[0], error[0], save_evaluation_to,
                             epoch_number, is_final, device)

    if no_pbar and rank == 0:
        print(f'{description}:'
              f'\n\tleft ssim: {average_left_ssim:.2f}'
              f'\n\tright ssim: {average_right_ssim:.2f}'
              f'\n\tause: {average_ause:.2f}'
              f'\n\taurg: {average_aurg:.2f}'
              f'\n\tdisparity scale: {scale:.2f}')

    average_ssim = (average_left_ssim, average_right_ssim)
    average_spars = (average_ause, average_aurg)

    return average_ssim, average_spars
