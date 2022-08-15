import re
import glob
import os.path

from typing import Dict, List, Optional
from PIL import Image, ImageFile

from torch import Tensor
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


class CityScapesDataset(Dataset):
    """Dataset class for loading CityScapes 8-bit images.

    Given the root of the dataset path, this class will find all left and
    right `.png` images and collect each pair as a dictionary of Tensors.

    If there are any missing image IDs from either left or right folders,
    the pair is ignored.

    Note:
        Transforms must be able to handle dictionaries containing left and
        right views as separate entries.
    """
    LEFT_PATH = 'leftImg8bit'
    RIGHT_PATH = 'rightImg8bit'
    EXTENSION = 'png'

    FILENAME_REGEX = re.compile(r'([a-z]+_\d+_\d+)_(\w+)\.(\w+)')

    def __init__(self, root: str, split: str,
                 transform: Optional[object] = None,
                 limit: Optional[int] = None) -> None:

        if split not in ('train', 'val', 'test'):
            raise ValueError('Split must be either "train", "val" or "test".')

        left_glob = os.path.join(root, self.LEFT_PATH, split,
                                 '**', f'*.{self.EXTENSION}')

        right_glob = os.path.join(root, self.RIGHT_PATH, split,
                                  '**', f'*.{self.EXTENSION}')

        left_images = glob.glob(left_glob)
        right_images = glob.glob(right_glob)

        left_ids = set(self.image_ids(left_images))
        right_ids = set(self.image_ids(right_images))

        missing = left_ids.symmetric_difference(right_ids)

        if len(missing) > 0:
            print(f'Missing {len(missing):,} images from the dataset.')
            left_images = [i for i in left_images if i not in missing]
            right_images = [i for i in right_images if i not in missing]
            print(f'Dataset reduced to {len(left_images):,} images.')

        self.lefts = sorted(left_images[:limit])
        self.rights = sorted(right_images[:limit])

        self.transform = transform

    def image_ids(self, image_paths: List[str]) -> List[str]:
        """Retrieve image ids given all their basenames.

        If a match cannot be made, the image basename is ignored.

        Args:
            image_paths (List[str]): The list of image basenames.
        Returns:
            List[str]: A list of image IDs in the order they were passed.
        """
        basenames = map(os.path.basename, image_paths)
        matches = map(self.FILENAME_REGEX.match, basenames)

        return [m.group(1) for m in matches if m is not None]

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

        left = Image.open(left_path).convert('RGB')
        right = Image.open(right_path).convert('RGB')

        image_pair = {'left': left, 'right': right}

        if self.transform is not None:
            image_pair = self.transform(image_pair)

        return image_pair

    def __len__(self) -> int:
        return len(self.lefts)
