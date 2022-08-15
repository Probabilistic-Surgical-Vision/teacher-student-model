import torch.nn as nn
from torch import Tensor

from .decoder import DepthDecoder, DecoderOut
from .encoder import RandomEncoder


class RandomlyConnectedModel(nn.Module):
    """The full architecture of the randomly-connected model.

    Args:
        encoder (dict): The config for the encoder architecture.
        decoder (dict): The config for the decoder architecture.
    """
    def __init__(self, encoder: dict, decoder: dict) -> None:
        super().__init__()

        self.encoder = RandomEncoder(**encoder)
        self.decoder = DepthDecoder(**decoder)

    def forward(self, image: Tensor, scale: float = 1) -> DecoderOut:
        encodings = self.encoder(image)
        return self.decoder(image, *encodings, scale=scale)
