import os.path
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader

from torchvision.utils import save_image

import tqdm

from . import utils as u
from .utils import Device


def save_comparisons(image: Tensor, ensemble: Tensor, prediction: Tensor,
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
    pred_disp, pred_error = torch.split(prediction, [2, 2], dim=0)
    ensemble_disp, ensemble_error = torch.split(ensemble, [2, 2], dim=0)

    left_error, right_error = torch.split(error, [3, 3], dim=0)
    error = torch.cat((left_error.mean(0, True), right_error.mean(0, True)))

    prediction_image = u.get_comparison(image, pred_disp, pred_error,
                                        scale=False, device=device)
    disparity_image = u.get_comparison(image, ensemble_disp, pred_disp,
                                       recon, scale=True, device=device)
    error_image = u.get_comparison(image, ensemble_error, pred_error,
                                   error, scale=True, device=device)

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
    save_image(error_image, uncertainty_filename)


@torch.no_grad()
def evaluate_model(model: Module, loader: DataLoader,
                   loss_function: Module, scale: float = 1.0,
                   disc: Optional[Module] = None,
                   disc_loss_function: Optional[Module] = None,
                   save_evaluation_to: Optional[str] = None,
                   epoch_number: Optional[int] = None,
                   is_final: bool = True,
                   scales: int = 4, device: Device = 'cpu',
                   no_pbar: bool = False,
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
    running_disp_loss = 0
    running_error_loss = 0
    running_disc_loss = 0

    disp_loss_per_image = None
    error_loss_per_image = None
    disc_loss_per_image = None

    batch_size = loader.batch_size \
        if loader.batch_size is not None \
        else len(loader)

    description = 'Evaluation'
    tepoch = tqdm.tqdm(loader, description, unit='batch',
                       disable=(no_pbar or rank > 0))

    for i, image_pair in enumerate(tepoch):
        left = image_pair['left'].to(device)
        right = image_pair['right'].to(device)
        ensemble = image_pair['ensemble'].to(device)

        images = torch.cat([left, right], dim=1)

        image_pyramid = u.scale_pyramid(images, scales)
        ensemble_pyramid = u.scale_pyramid(ensemble, scales)

        disparities = model(left, scale)

        recon_pyramid = u.reconstruct_pyramid(disparities, image_pyramid)
        disp_loss, error_loss = loss_function(image_pyramid, disparities,
                                              ensemble_pyramid,
                                              recon_pyramid, i, disc)

        if disc is not None:
            disc_loss = u.run_discriminator(image_pyramid, recon_pyramid,
                                            disc, disc_loss_function,
                                            batch_size)

        if rank > 0:
            continue

        running_disp_loss += disp_loss.item()
        running_error_loss += error_loss.item()

        disp_loss_per_image = running_disp_loss / ((i+1) * batch_size)
        error_loss_per_image = running_error_loss / ((i+1) * batch_size)

        if disc is not None:
            running_disc_loss += disc_loss.item()
            disc_loss_per_image = running_disc_loss / ((i+1) * batch_size)

        tepoch.set_postfix(disp=disp_loss_per_image,
                           error=error_loss_per_image,
                           disc=disc_loss_per_image,
                           scale=scale)

        if save_evaluation_to is not None and i == 0:
            original_image = images[0]
            ensemble_image = ensemble[0]

            disparity_image = disparities[0][0]
            recon_image = recon_pyramid[0][0]

            error_image = loss_function.reprojection_error[0]

            save_comparisons(original_image, ensemble_image,
                             disparity_image, recon_image,
                             error_image, save_evaluation_to,
                             epoch_number, is_final, device)

    if no_pbar and rank == 0:
        disc_loss_string = f'{disc_loss_per_image:.2e}' \
            if disc_loss_per_image is not None else None

        print(f'{description}:'
              f'\n\tdisparity loss: {disp_loss_per_image:.2e}'
              f'\n\terror loss: {error_loss_per_image:.2e}'
              f'\n\tdiscriminator loss: {disc_loss_string}'
              f'\n\tdisparity scale: {scale:.2f}')

    return disp_loss_per_image, error_loss_per_image, disc_loss_per_image
