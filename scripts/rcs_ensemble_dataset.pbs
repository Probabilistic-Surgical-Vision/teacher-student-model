#!/bin/bash

#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=4:ngpus=1:mem=50gb:gpu_type=RTX6000

cd $PBS_O_WORKDIR

module load anaconda3/personal
source activate ensemble-env

python ensemble_dataset.py ensemble_models/ da-vinci \
    --save-to davinci-ensemble/ --home ../ --batch-size 4
