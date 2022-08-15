from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from .layers.encoder import EncoderStage

KernelSize = Union[int, Tuple[int, int]]
ImagePyramid = List[Tensor]


class RandomDiscriminator(nn.Module):
    """The full discriminator module.

    Note:
        If `load_graph` is specified, it will override all parameters for
        building graphs.

    Args:
        layers (List[dict]): A list of configs for each encoder stage. These
            are unpacked and passed as kwargs to each stage.
        final_conv (dict): The config for the final convolutional layer.
        linear_in_features (int): The number of features from the final
            convolutional layer when flattened.
        load_graph (Optional[str], optional): The path to a directory
            containing all graphs for each stage in a `gpickle` format.
            Defaults to None.
        nodes (int, optional): The number of nodes per graph. Defaults to 5.
        seed (Optional[int], optional): The random seed for building new
            graphs. Defaults to 42.
    """
    def __init__(self, layers: List[dict], final_conv: dict,
                 linear_in_features: int, load_graph: Optional[str] = None,
                 nodes: int = 5, seed: int = 42) -> None:

        super().__init__()

        self.layers = nn.ModuleList()

        for i, layer_config in enumerate(layers):
            self.layers.append(EncoderStage(**layer_config, stage=(i+1),
                                            nodes=nodes, seed=seed,
                                            load_graph=load_graph))

        self.conv = EncoderStage(**final_conv,
                                 stage=(len(self.layers)+1),
                                 nodes=nodes, seed=seed,
                                 load_graph=load_graph)

        self.linear = nn.Linear(linear_in_features, 1)

    def features(self, pyramid: ImagePyramid) -> ImagePyramid:
        """Get the pyramid of feature maps from the encoder stages.

        Args:
            pyramid (ImagePyramid): The original image at different scales for
                each encoder stage.

        Returns:
            ImagePyramid: The feature maps with the same shape as the original
                image pyramid.
        """
        features = []

        pyramid_iterator = zip(pyramid, self.layers)
        for i, (images, layer) in enumerate(pyramid_iterator):
            if i == 0:
                out = layer(images)
            else:
                feature_concat = torch.cat((out, images), dim=1)
                out = layer(feature_concat)

            features.append(out)

        return features

    def forward(self, pyramid: ImagePyramid) -> Tensor:
        feature = self.features(pyramid)[-1]

        out = self.conv(feature)
        out = out.view(out.size(0), -1)

        out = self.linear(out)

        return torch.sigmoid(out)
