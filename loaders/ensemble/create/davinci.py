import os
import os.path
import glob

from collections import OrderedDict
from typing import Optional, Union

import torch
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader

import tqdm

from ...davinci import DaVinciDataset

Device = Union[str, torch.device]


class CreateDaVinciEnsembleDataset(Dataset):

    def __init__(self, models_path: str, davinci_path: str,
                 split: str = 'train', batch_size: int = 8,
                 transform: Optional[object] = None) -> None:

        self.model_states = []
        self.dataset_path = davinci_path
        self.batch_size = batch_size

        models_glob = os.path.join(models_path, '*.pt')

        for model_path in glob.glob(models_glob):
            model_state = self.prepare_state_dict(model_path)

            self.model_states.append(model_state)
    
        self.dataset = DaVinciDataset(davinci_path, split, transform, limit=32)
        self.dataloader = DataLoader(self.dataset, batch_size, shuffle=False)
        
        print(f'Size of da Vinci Dataset: {len(self.dataset):,}')
    
    def prepare_state_dict(self, model_path: str) -> OrderedDict:
        state_dict = torch.load(model_path)
        return {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    def get_ensemble_predictions(self, save_to: str, device: Device) -> None:

        etimations_glob = os.path.join(save_to, '*.pt')
        estimations_paths = glob.glob(etimations_glob)

        description = f'Calculating Mean and Variance'
        tepoch = tqdm.tqdm(estimations_paths, description, unit='prediction')

        for estimations_path in tepoch:
            estimations = torch.load(estimations_path, map_location=device)
            
            mean = estimations.mean(dim=0)
            variance = estimations.var(dim=0)
            
            print(variance)

            combined = torch.cat((mean, variance), dim=0)
            
            print(estimations.shape, combined.shape)
            
            torch.save(combined, estimations_path)

    def get_model_predictions(self, model: Module, model_number: int,
                              save_to: str, device: Device) -> None:
        description = f'Model #{model_number}'
        tepoch = tqdm.tqdm(self.dataloader, description, unit='batch')

        for i, image_pair in enumerate(tepoch):
            left = image_pair['left'].to(device)
            predictions = model(left)

            for j, estimation in enumerate(predictions):
                estimation = estimation.unsqueeze(0)
                image_id = (self.batch_size * i) + j + 1
                filename = f'ensemble_{image_id:04}.pt'
                filepath = os.path.join(save_to, filename)

                if os.path.isfile(filepath):
                    prev_estimations = torch.load(filepath,
                                                  map_location=device)
                    estimation = torch.cat((prev_estimations, estimation))

                torch.save(estimation, filepath)

    @torch.no_grad()
    def create(self, blank_model: Module, save_to: Optional[str],
               device: Device = 'cpu') -> None:

        if save_to is None or save_to == self.dataset_path:
            save_to = os.path.join(self.dataset_path, 'ensemble')
            print(f'Saving predictions to:\n\t{save_to}')

        os.makedirs(save_to, exist_ok=True)

        model = blank_model.to(device)
        model.eval()

        for i, state_dict in enumerate(self.model_states):
            model.load_state_dict(state_dict)
            self.get_model_predictions(model, (i+1), save_to, device)

        self.get_ensemble_predictions(save_to, device)