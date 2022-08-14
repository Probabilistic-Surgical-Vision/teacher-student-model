import argparse
import os
import os.path

import torch
from torchvision import transforms

import yaml

from model import RandomlyConnectedModel
from loaders.ensemble import create

import train

parser = argparse.ArgumentParser()

parser.add_argument('models', type=str,
                    help='The path to the models to use as an ensemble.')
parser.add_argument('dataset', choices=['da-vinci', 'scared'], type=str,
                    help='The dataset to generate ensemble predictions for.')
parser.add_argument('--save-to', '-s', type=str, default=None,
                    help='The location to save the ensemble predictions to.')
parser.add_argument('--batch-size', '-b', default=8, type=int,
                    help='The batch size to train/evaluate the model with.')
parser.add_argument('--split', choices=['train', 'test'], default='train',
                    help='The dataset to generate ensemble predictions for.')
parser.add_argument('--workers', '-w', default=8, type=int,
                    help='The number of workers to use for the dataloader.')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Prevent program from training model using cuda.')
parser.add_argument('--home', default=os.environ['HOME'], type=str,
                    help='Override the home directory (to find datasets).')


def main(args: argparse.Namespace) -> None:
    print("Arguments passed:")
    for key, value in vars(args).items():
        print(f'\t- {key}: {value}')

    creator_class = create.CreateDaVinciEnsembleDataset \
        if args.dataset == 'da-vinci' \
        else create.CreateSCAREDEnsembleDataset

    dataset_path = os.path.join(args.home, 'datasets', args.dataset)

    transform = transforms.Compose([
        train.transforms.ResizeImage((256, 512)),
        train.transforms.ToTensor()])

    creator = creator_class(args.models, dataset_path,
                            args.split, args.batch_size,
                            transform, args.workers)

    ensemble_config_path = os.path.join(args.models, 'config.yml')
    with open(ensemble_config_path) as f:
        ensemble_config = yaml.load(f, Loader=yaml.Loader)

    
    device = torch.device('cuda') if torch.cuda.is_available() \
        and not args.no_cuda else torch.device('cpu')
    
    blank_model = RandomlyConnectedModel(**ensemble_config)

    creator.create(blank_model, args.save_to, device)

    print(f'Ensemble dataset for "{args.dataset}" completed.')


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)