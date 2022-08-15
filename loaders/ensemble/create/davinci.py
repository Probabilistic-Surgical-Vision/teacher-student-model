import os
import os.path
import glob

from collections import OrderedDict
from typing import Optional, Union

import numpy as np

import tifffile

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader

import tqdm

from ...davinci import DaVinciDataset

Device = Union[str, torch.device]


class CreateDaVinciEnsembleDataset(Dataset):

    def __init__(self, models_path: str, davinci_path: str,
                 split: str = 'train', batch_size: int = 8,
                 transform: Optional[object] = None,
                 workers: int = 8) -> None:

        self.model_states = []
        self.dataset_path = davinci_path
        self.batch_size = batch_size

        models_glob = os.path.join(models_path, '*.pt')

        for model_path in glob.glob(models_glob):
            model_state = self.prepare_state_dict(model_path)

            self.model_states.append(model_state)
    
        self.dataset = DaVinciDataset(davinci_path, split, transform)
        self.dataloader = DataLoader(self.dataset, batch_size,
                                     shuffle=False, num_workers=workers)
        
        print(f'Size of da Vinci Dataset: {len(self.dataset):,}')
    
    def prepare_state_dict(self, model_path: str) -> OrderedDict:
        state_dict = torch.load(model_path)
        return {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    torch.no_grad()
    def ensemble_predict(self, image: Tensor, model: Module) -> Tensor:

        predictions = []

        for state_dict in self.model_states:
            model.load_state_dict(state_dict)
            prediction = model(image)

            predictions.append(prediction)
        
        predictions = torch.stack(predictions)
        mean = predictions.mean(dim=0)
        variance = predictions.var(dim=0)

        return torch.cat((mean, variance), dim=1)

    @torch.no_grad()
    def create(self, blank_model: Module, save_to: Optional[str],
               device: Device = 'cpu') -> None:

        if save_to is None or save_to == self.dataset_path:
            save_to = os.path.join(self.dataset_path, 'ensemble')
            print(f'Saving predictions to:\n\t{save_to}')

        os.makedirs(save_to, exist_ok=True)

        model = blank_model.to(device)
        model.eval()

        tepoch = tqdm.tqdm(self.dataloader, unit='batch')

        for i, image_pair in enumerate(tepoch):
            left = image_pair['left'].to(device)
            estimations = self.ensemble_predict(left, model)

            for j, estimation in enumerate(estimations):
                image_id = (self.batch_size * i) + j + 1
                filename = f'ensemble_{image_id:04}.tiff'
                filepath = os.path.join(save_to, filename)
                
                estimation = estimation.cpu().numpy() \
                    .astype(np.float32)

                tifffile.imwrite(filepath, estimation)