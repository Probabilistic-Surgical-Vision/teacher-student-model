from typing import Dict, Tuple

from numpy import random

import torch
import torch.nn.functional as F
from torch import Tensor

from torchvision import transforms

ImageDict = Dict[str, Tensor]
BoundsTuple = Tuple[float, float]
ImageSizeTuple = Tuple[int, int]


class ResizeImage:

    def __init__(self, size: ImageSizeTuple = (256, 512)) -> None:
        self.size = size
        self.transform = transforms.Resize(size)

    def __call__(self, image_pair: ImageDict) -> ImageDict:
        image_pair['left'] = self.transform(image_pair['left'])
        image_pair['right'] = self.transform(image_pair['right'])

        if 'ensemble' in image_pair:
            ensemble = F.interpolate(image_pair['ensemble'], self.size,
                                     mode='bilinear', align_corners=True)

            image_pair['ensemble'] = ensemble

        return image_pair


class ToTensor:

    def __init__(self) -> None:
        self.transform = transforms.ToTensor()

    def __call__(self, image_pair: ImageDict) -> ImageDict:
        image_pair['left'] = self.transform(image_pair['left'])
        image_pair['right'] = self.transform(image_pair['right'])

        return image_pair


class RandomFlip:

    def __init__(self, p: float = 0.5) -> None:
        self.probability = p
        self.transform = transforms.RandomHorizontalFlip(1)

    def __call__(self, image_pair: ImageDict) -> ImageDict:
        if random.random() < self.probability:
            image_pair['left'] = self.transform(image_pair['left'])
            image_pair['right'] = self.transform(image_pair['right'])

            if 'ensemble' in image_pair:
                image_pair['ensemble'] = image_pair['ensemble'][:, :, ::-1]

        return image_pair


class RandomAugment:

    def __init__(self, p: float, gamma: BoundsTuple,
                 brightness: BoundsTuple, colour: BoundsTuple) -> None:

        self.probability = p

        self.gamma = gamma
        self.brightness = brightness
        self.colour = colour

    def shift_gamma(self, x: Tensor, gamma: torch.float) -> Tensor:
        return x ** gamma

    def shift_brightness(self, x: Tensor, brightness: torch.float) -> Tensor:
        return x * brightness

    def shift_colour(self, x: Tensor, colour: Tensor) -> Tensor:
        return x * colour.unsqueeze(-1).unsqueeze(-1)

    def transform(self, x: Tensor, gamma: torch.float, brightness:
                  torch.float, colour: Tensor) -> Tensor:

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
