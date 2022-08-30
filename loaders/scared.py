import glob
import os.path

from typing import Dict, Optional
from PIL import Image, ImageFile

import torch
from torch import Tensor
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


class SCAREDEnsembleDataset(Dataset):
    """Dataset class for loading the SCARED images.

    Given the root of the dataset path, this class will find all left and
    right `.png` images and collect each pair as a dictionary of Tensors.

    If there are any missing image IDs from either left or right folders,
    the pair is ignored.

    Note:
        Transforms must be able to handle dictionaries containing left and
        right views as separate entries.

    Args:
        root (str): Path to the root of the dataset directory.
        split (str): The folder in the dataset to use. Must be "train" or
            "test".
        transform (Optional[object], optional): The transforms to apply to
            each image pair while loading. Defaults to None.
        limit (Optional[int], optional): The maximum number of images to load.
            Loads all images if None. Defaults to None.
    """
    LEFT_PATH = 'left'
    RIGHT_PATH = 'right'
    IMAGE_EXT = 'png'
    ENSEMBLE_EXT = 'tiff'

    def __init__(self, scared_path: str, ensemble_path: str, split: str,
                 transform: Optional[object] = None,
                 limit: Optional[int] = None) -> None:

        if split not in ('train', 'test'):
            raise ValueError('Split must be either "train" or "test".')

        left_glob = os.path.join(scared_path, split, '**',
                                 self.LEFT_PATH, f'*.{self.IMAGE_EXT}')
        right_glob = os.path.join(scared_path, split, '**',
                                  self.RIGHT_PATH, f'*.{self.IMAGE_EXT}')
        ensemble_glob = os.path.join(ensemble_path, split,
                                     f'*.{self.ENSEMBLE_EXT}')

        left_images = glob.glob(left_glob)
        right_images = glob.glob(right_glob)

        ensemble_estimations = glob.glob(ensemble_glob)

        left_names = set(map(os.path.basename, left_images))
        right_names = set(map(os.path.basename, right_images))

        missing = left_names.symmetric_difference(right_names)

        if len(missing) > 0:
            print(f'Missing {len(missing):,} images from the dataset.')
            left_images = [i for i in left_images if i not in missing]
            right_images = [i for i in right_images if i not in missing]
            print(f'Dataset reduced to {len(left_images):,} images.')
        elif len(ensemble_estimations) != len(left_images):
            raise Exception(f'Ensemble predictions and'
                            ' image numbers do not match.')

        self.lefts = sorted(left_images[:limit])
        self.rights = sorted(right_images[:limit])

        self.ensembles = sorted(ensemble_estimations[:limit])

        self.transform = transform

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        """Retrieve a single sample from the dataset.

        Args:
            idx (int): The index of the sample in the dataset.

        Returns:
            Dict[str, Tensor]: The left and right images packaged as a
                dictionary containing `left` and `right` keys.
        """
        left_path = self.lefts[idx]
        right_path = self.rights[idx]

        ensemble_path = self.ensembles[idx]

        left = Image.open(left_path).convert('RGB')
        right = Image.open(right_path).convert('RGB')

        ensemble_prediction = torch.open(ensemble_path)
        
        image_pair = {
            'left': left, 'right': right,
            'ensemble': ensemble_prediction
        }

        if self.transform is not None:
            image_pair = self.transform(image_pair)

        return image_pair

    def __len__(self) -> int:
        return len(self.lefts)