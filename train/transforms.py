from typing import Dict, Tuple

from numpy import random

import torch
import torch.nn.functional as F
from torch import Tensor

from torchvision import transforms

ImageDict = Dict[str, Tensor]
BoundsTuple = Tuple[float, float]
ImageSize = Tuple[int, int]


class ResizeImage:
    """Resize the stereo images grouped in a dictionary.

    Args:
        size (ImageSize, optional): The target image size. Defaults to
            (256, 512).
    """
    def __init__(self, size: ImageSize = (256, 512)) -> None:
        self.transform = transforms.Resize(size)
        self.size = size

    def __call__(self, image_pair: ImageDict) -> ImageDict:
        image_pair['left'] = self.transform(image_pair['left'])
        image_pair['right'] = self.transform(image_pair['right'])

        if 'ensemble' in image_pair:
            ensemble = image_pair['ensemble'].unsqueeze(0)
            ensemble = F.interpolate(ensemble, self.size, mode='bilinear',
                                     align_corners=True)

            image_pair['ensemble'] = ensemble.squeeze(0)

        return image_pair


class ToTensor:
    """Convert stereo PIL images grouped in a dictionary."""
    def __init__(self) -> None:
        self.transform = transforms.ToTensor()

    def __call__(self, image_pair: ImageDict) -> ImageDict:
        image_pair['left'] = self.transform(image_pair['left'])
        image_pair['right'] = self.transform(image_pair['right'])

        return image_pair


class RandomFlip:
    """Random horizontal flip stereo images grouped in a dictionary.

    Args:
        p (float, optional): The probabaility of a horizontal flip. Defaults
            to 0.5.
    """
    def __init__(self, p: float = 0.5) -> None:
        self.probability = p
        self.transform = transforms.RandomHorizontalFlip(1)

    def __call__(self, image_pair: ImageDict) -> ImageDict:
        if random.random() < self.probability:
            image_pair['left'] = self.transform(image_pair['left'])
            image_pair['right'] = self.transform(image_pair['right'])

            if 'ensemble' in image_pair:
                image_pair['ensemble'] = image_pair['ensemble'].flip(2)

        return image_pair


class RandomAugment:
    """Randomly alter the brightness, contrast and colour of stereo images
    grouped in a dictionary.

    Args:
        p (float): The probability of augmentation.
        gamma (BoundsTuple): The range of exponents to randomly sample for
            contrast.
        brightness (BoundsTuple): The range of multipliers to randomly sample
            for brightness.
        colour (BoundsTuple): The range of multipliers to randomly sample for
            channel-specific brightness (i.e. altering the colour).
    """
    def __init__(self, p: float, gamma: BoundsTuple,
                 brightness: BoundsTuple, colour: BoundsTuple) -> None:

        self.probability = p

        self.gamma = gamma
        self.brightness = brightness
        self.colour = colour

    def shift_gamma(self, x: Tensor, gamma: torch.float) -> Tensor:
        """Alter contrast by exponentiating each pixel value."""
        return x ** gamma

    def shift_brightness(self, x: Tensor, brightness: torch.float) -> Tensor:
        """Alter brightness by multiplying each pixel value."""
        return x * brightness

    def shift_colour(self, x: Tensor, colour: Tensor) -> Tensor:
        """Alter colour by multiplying each pixel channel value."""
        return x * colour.unsqueeze(-1).unsqueeze(-1)

    def transform(self, x: Tensor, gamma: torch.float, brightness:
                  torch.float, colour: Tensor) -> Tensor:
        """Apply the augmentation to a single image.

        Args:
            x (Tensor): The image to augment.
            gamma (torch.float): The exponent for altering contrast.
            brightness (torch.float): The multiplier for altering brightness.
            colour (Tensor): An array of multipliers for altering the
                brightness of each channel (must have a length of 3).

        Returns:
            Tensor: The augmented image.
        """
        x = self.shift_gamma(x, gamma)
        x = self.shift_brightness(x, brightness)
        x = self.shift_colour(x, colour)

        return torch.clamp(x, 0, 1)

    def __call__(self, image_pair: ImageDict) -> ImageDict:
        left, right = image_pair['left'], image_pair['right']

        if random.random() < self.probability:
            g = random.uniform(*self.gamma)
            b = random.uniform(*self.brightness)
            c = torch.tensor(random.uniform(*self.colour, 3),
                             dtype=torch.float)

            left = self.transform(left, g, b, c)
            right = self.transform(right, g, b, c)

            image_pair['left'] = left
            image_pair['right'] = right

        return image_pair
