import os
import os.path
import glob

from collections import OrderedDict
from typing import Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader

import tqdm

from ...scared import SCAREDDataset

Device = Union[str, torch.device]


class CreatescaredEnsembleDataset(Dataset):

    def __init__(self, models_path: str, scared_path: str,
                 split: str = 'train', batch_size: int = 8,
                 transform: Optional[object] = None) -> None:

        self.model_states = []
        self.dataset_path = scared_path
        self.batch_size = batch_size

        models_glob = os.path.join(models_path, '*.pt')

        for model_path in glob.glob(models_glob):
            model_state = self.prepare_state_dict(model_path)

            self.model_states.append(model_state)
    
        self.dataset = SCAREDDataset(scared_path, split, transform)
        self.dataloader = DataLoader(self.dataset, batch_size, shuffle=False)
        
        print(f'Size of SCARED Dataset: {len(self.dataset):,}')
    
    def prepare_state_dict(self, model_path: str) -> OrderedDict:
        state_dict = torch.load(model_path)
        return {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    def ensemble_predict(self, image: Tensor,
                         model: Module) -> Tuple[Tensor, Tensor]:

        predictions = []

        for state_dict in self.model_states:
            model.load_state_dict(state_dict)

            prediction = model(image)
            predictions.append(prediction)
                
        predictions = torch.stack(predictions)
        mean = predictions.mean(dim=0)
        variance = predictions.var(dim=0)

        combined = torch.cat((mean, variance), dim=1)

        return torch.split(combined, 1, dim=0)

    @torch.no_grad()
    def create(self, blank_model: Module, save_to: Optional[str],
               device: Device = 'cpu') -> None:

        if save_to is None or save_to == self.dataset_path:
            save_to = os.path.join(self.dataset_path, 'ensemble')
            print(f'Saving predictions to:\n\t{save_to}')

        os.makedirs(save_to, exist_ok=True)
        tepoch = tqdm.tqdm(self.dataloader, unit='batch')

        model = blank_model.to(device)
        model.eval()

        for i, image_pair in enumerate(tepoch):
            left = image_pair['left'].to(device)
            estimations = self.ensemble_predict(left, model, device)

            for j, estimation in enumerate(estimations):
                estimation = estimation.squeeze()

                image_id = (self.batch_size * i) + j + 1
                filename = f'ensemble_{image_id:04}.pt'
                filepath = os.path.join(save_to, filename)

                torch.save(estimation, filepath)
