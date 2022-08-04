#!/bin/bash

#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=16:ngpus=4:mem=125gb:gpu_type=RTX6000

cd $PBS_O_WORKDIR

module load anaconda3/personal
source activate tukra-env

python parallel_main.py config.yml da-vinci -b 8 -e 200 -lr 0.0004 \
    --save-model-to trained/da-vinci --save-model-every 10 \
    --save-evaluation-to results/da-vinci --evaluate-every 10 \
    --home /rds/general/user/lem3617/home --number-of-gpus 4