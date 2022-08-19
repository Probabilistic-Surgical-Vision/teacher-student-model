import glob
import os.path
from os.path import basename
from typing import Dict, Optional

from PIL import Image, ImageFile

import tifffile

import torch
from torch import Tensor
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


class DaVinciEnsembleDataset(Dataset):
    """Dataset class for loading the Hamlyn da Vinci images.

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
    LEFT_PATH = 'image_0'
    RIGHT_PATH = 'image_1'
    IMAGE_EXT = 'png'
    ENSEMBLE_EXT = 'tiff'

    def __init__(self, davinci_path: str, ensemble_path: str, split: str,
                 transform: Optional[object] = None,
                 limit: Optional[int] = None) -> None:

        if split not in ('train', 'test'):
            raise ValueError('Split must be either "train" or "test".')

        left_glob = os.path.join(davinci_path, split, self.LEFT_PATH,
                                 f'*.{self.IMAGE_EXT}')

        right_glob = os.path.join(davinci_path, split, self.RIGHT_PATH,
                                  f'*.{self.IMAGE_EXT}')

        ensemble_glob = os.path.join(ensemble_path, split,
                                     f'*.{self.ENSEMBLE_EXT}')

        left = glob.glob(left_glob)
        right = glob.glob(right_glob)

        ensemble = glob.glob(ensemble_glob)

        left_names = set(map(self.name, left))
        right_names = set(map(self.name, right))
        ensemble_names = set(map(self.name, ensemble))

        stereo_missing = left_names.symmetric_difference(right_names)
        ensemble_missing = left_names.symmetric_difference(ensemble_names)

        missing = set.union(stereo_missing, ensemble_missing)

        if len(missing) > 0:
            print(f'Missing {len(missing):,} images from the dataset.')
            left = [i for i in left if self.name(i) not in missing]
            right = [i for i in right if self.name(i) not in missing]
            ensemble = [i for i in ensemble if self.name(i) not in missing]
            print(f'Dataset reduced to {len(left):,} images.')

        self.lefts = sorted(left[:limit])
        self.rights = sorted(right[:limit])

        self.ensembles = sorted(ensemble[:limit])

        self.transform = transform

    @staticmethod
    def name(path: str) -> str:
        return os.path.splitext(basename(path))[0]

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

        ensemble_prediction = tifffile.imread(ensemble_path)
        
        image_pair = {
            'left': left, 'right': right,
            'ensemble': torch.from_numpy(ensemble_prediction)
        }

        if self.transform is not None:
            image_pair = self.transform(image_pair)

        return image_pair

    def __len__(self) -> int:
        return len(self.lefts)
