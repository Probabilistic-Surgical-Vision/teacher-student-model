import os
import os.path
from copy import deepcopy
from typing import List, Optional, Tuple

import torch
from torch.nn import Module
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

import tqdm

from .evaluate import evaluate_model

from . import utils as u
from .utils import Device

Loss = List[float]


def save_model(model: Module, save_model_to: str,
               disc: Optional[Module] = None,
               epoch_number: Optional[int] = None,
               is_final: bool = False) -> None:
    """Save the model state dict as a `.pt` file.

    Args:
        model (Module): The model to save.
        save_model_to (str): The directory to save the model to.
        disc (Optional[Module], optional): The discriminator to save (if
            applicable). Defaults to None.
        epoch_number (Optional[int], optional): The training epoch (if
            applicable). Defaults to None.
        is_final (bool, optional): Flag for saving model after training.
            Defaults to False.
    """
    os.makedirs(save_model_to, exist_ok=True)

    filename = 'final.pt' if is_final else f'epoch_{epoch_number:03}.pt'
    filepath = os.path.join(save_model_to, filename)

    if disc is not None:
        state_dict = {
            'model': model.state_dict(),
            'disc': disc.state_dict()
        }
    else:
        state_dict = model.state_dict()

    print(f'Saving model to:\n\t{filepath}')
    torch.save(state_dict, filepath)


def train_one_epoch(model: Module, loader: DataLoader, loss_function: Module,
                    model_optimiser: Optimizer, scale: float,
                    disc: Optional[Module] = None,
                    disc_optimiser: Optional[Optimizer] = None,
                    disc_loss_function: Optional[Module] = None,
                    epoch_number: Optional[int] = None,
                    scales: int = 4, perceptual_update_freq: int = 10,
                    device: Device = 'cpu', no_pbar: bool = False,
                    rank: int = 0) -> Tuple[float, float]:
    """Train the model for a single epoch.

    Args:
        model (Module): The model to train.
        loader (DataLoader): The dataloader to iterate through.
        loss_function (Module): The model loss function.
        model_optimiser (Optimizer): The model optimiser.
        scale (float): A multiplier to scale the disparity prediction by.
        disc (Optional[Module], optional): The discriminator. Defaults to
            None.
        disc_optimiser (Optional[Optimizer], optional): The discriminator
            optimiser. Defaults to None.
        disc_loss_function (Optional[Module], optional): The discriminator
            loss function. Defaults to None.
        epoch_number (Optional[int], optional): The training epoch (for
            saving models and the progress bar description). Defaults to
            None.
        scales (int, optional): The number of elements in the image pyramids.
            Defaults to 4.
        perceptual_update_freq (int, optional): The number of batches before
            updating the discriminator clone. Defaults to 10.
        device (Device, optional): The torch device. Defaults to 'cpu'.
        no_pbar (bool, optional): Disable the progress bar. Defaults to False.
        rank (int, optional): The global rank of the GPU (for
            DistributedDataParallel only). Defaults to 0.

    Returns:
        float: The average model loss per image.
        float: The average discriminator loss per image.
    """
    model.train()

    if disc is not None:
        disc.train()

    running_disp_loss = 0
    running_error_loss = 0
    running_disc_loss = 0

    disp_loss_per_image = None
    error_loss_per_image = None
    disc_loss_per_image = None

    batch_size = loader.batch_size \
        if loader.batch_size is not None else len(loader)
    description = f'Epoch #{epoch_number}' \
        if epoch_number is not None else 'Epoch'
    disc_clone = deepcopy(disc) if disc is not None else None

    tepoch = tqdm.tqdm(loader, description, unit='batch',
                       disable=(no_pbar or rank > 0))

    for i, image_pair in enumerate(tepoch):

        left = image_pair['left'].to(device)
        right = image_pair['right'].to(device)
        truth = image_pair['ensemble'].to(device)

        images = torch.cat([left, right], dim=1)

        image_pyramid = u.scale_pyramid(images, scales)
        truth_pyramid = u.scale_pyramid(truth, scales)

        model_optimiser.zero_grad()
        disparities = model(left, scale)

        disp_loss, error_loss = loss_function(image_pyramid, disparities,
                                              truth_pyramid, i, disc_clone)

        model_loss = disp_loss + error_loss

        model_loss.backward()
        model_optimiser.step()

        if rank == 0:
            running_disp_loss += disp_loss.item()
            running_error_loss += error_loss.item()

            disp_loss_per_image = running_disp_loss / ((i+1) * batch_size)
            error_loss_per_image = running_error_loss / ((i+1) * batch_size)

        if disc is not None:
            disc_optimiser.zero_grad()
            disc_loss = u.run_discriminator(image_pyramid, truth_pyramid,
                                            disc, disc_loss_function,
                                            batch_size)

            disc_loss.backward()
            disc_optimiser.step()

            if rank == 0:
                running_disc_loss += disc_loss.item()
                disc_loss_per_image = running_disc_loss / ((i+1) * batch_size)

        if disc is not None and i % perceptual_update_freq == 0:
            disc_clone.load_state_dict(disc.state_dict())

        if rank == 0:
            tepoch.set_postfix(disp=disp_loss_per_image,
                               error=error_loss_per_image,
                               disc=disc_loss_per_image,
                               scale=scale)

    if no_pbar and rank == 0:
        disc_loss_string = f'{disc_loss_per_image:.2e}' \
            if disc_loss_per_image is not None else None

        print(f'{description}:'
              f'\n\tdisparity loss: {disp_loss_per_image:.2e}'
              f'\n\terror loss: {error_loss_per_image:.2e}'
              f'\n\tdiscriminator loss: {disc_loss_string}'
              f'\n\tdisparity scale: {scale:.2f}')

    return disp_loss_per_image, error_loss_per_image, disc_loss_per_image


def train_model(model: Module, loader: DataLoader, loss_function: Module,
                epochs: int, learning_rate: float,
                disc: Optional[Module] = None,
                disc_loss_function: Optional[Module] = None,
                scheduler_decay_rate: float = 0.1,
                scheduler_step_size: int = 15,
                perceptual_update_freq: int = 10,
                val_loader: Optional[DataLoader] = None,
                evaluate_every: Optional[int] = None,
                save_evaluation_to: Optional[str] = None,
                save_every: Optional[int] = None,
                save_model_to: Optional[str] = None,
                finetune: bool = False,
                device: Device = 'cpu',
                no_pbar: bool = False,
                rank: int = 0) -> Tuple[Loss, Loss]:
    """Train the model for multiple epochs.

    Args:
        model (Module): The model to train
        loader (DataLoader): The dataloader for training.
        loss_function (Module): The model loss function.
        epochs (int): The number of epochs to train for.
        learning_rate (float): The initial learning rate.
        disc (Optional[Module], optional): The discriminator. Defaults to
            None.
        disc_loss_function (Optional[Module], optional): The discriminator
            loss function. Defaults to None.
        scheduler_decay_rate (float, optional): The rate at which the
            learning rate is scaled every full step. Defaults to 0.1.
        scheduler_step_size (int, optional): The number of epochs per full
            step. Defaults to 15.
        perceptual_update_freq (int, optional): The number of batches before
            updating the discriminator clone. Defaults to 10.
        val_loader (Optional[DataLoader], optional): The dataloader for
            evaluation. Defaults to None.
        evaluate_every (Optional[int], optional): The number of epochs per
            evaluation. Defaults to None.
        save_evaluation_to (Optional[str], optional): The directory to save
            evaluation results to. Defaults to None.
        save_every (Optional[int], optional): The number of epochs between
            saving the model. Defaults to None.
        save_model_to (Optional[str], optional): The directory to save model
            state dicts to. Defaults to None.
        finetune (bool, optional): Indicates the model is being finetuned.
            Defaults to False.
        device (Device, optional): The torch device to use. Defaults to 'cpu'.
        no_pbar (bool, optional): Disable the progress bar. Defaults to False.
        rank (int, optional): The global rank of the GPU (for
            DistributedDataParallel only). Defaults to 0.

    Returns:
        List[float]: The average model losses at each epoch.
        List[float]: The average discriminator losses at each epoch.
    """
    model_optimiser = Adam(model.parameters(), learning_rate)
    disc_optimiser = Adam(disc.parameters(), learning_rate) \
        if disc is not None else None

    scheduler = StepLR(model_optimiser, scheduler_step_size,
                       scheduler_decay_rate)

    training_losses = []
    validation_losses = []

    for i in range(epochs):
        scale = 1 if finetune else u.adjust_disparity_scale(epoch=i)

        loss = train_one_epoch(model, loader, loss_function, model_optimiser,
                               scale, disc, disc_optimiser,
                               disc_loss_function, epoch_number=(i+1),
                               perceptual_update_freq=perceptual_update_freq,
                               device=device, no_pbar=no_pbar, rank=rank)

        if rank == 0:
            training_losses.append(loss)

        scheduler.step()

        if evaluate_every is not None and (i+1) % evaluate_every == 0:
            loss = evaluate_model(model, val_loader, loss_function, scale,
                                  disc, disc_loss_function,
                                  save_evaluation_to, epoch_number=(i+1),
                                  device=device, is_final=False,
                                  no_pbar=no_pbar, rank=rank)

            if rank == 0:
                validation_losses.append(loss)

        if save_every is not None and (i+1) % save_every == 0 and rank == 0:
            save_model(model, save_model_to, disc,
                       epoch_number=(i+1))

    if rank == 0:
        print('Training completed.')

    if save_model_to is not None and rank == 0:
        save_model(model, save_model_to, is_final=True)

    return training_losses, validation_losses
