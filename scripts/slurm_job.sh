#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lem3617
#SBATCH --output=/vol/bitbucket/lem3617/tukra-model/results/run-001.out
#SBATCH --error=/vol/bitbucket/lem3617/tukra-model/results/run-001.err

BITBUCKET_HOME=/vol/bitbucket/lem3617
REPO_DIR=$BITBUCKET_HOME/tukra-model

cd $REPO_DIR

export PATH=$REPO_DIR/venv/bin/:$PATH

source activate
source /vol/cuda/11.0.3-cudnn8.0.5.39/setup.sh

TERM=vt100
/usr/bin/nvidia-smi
uptime

python main.py config.yml da-vinci -b 8 -e 200 -lr 0.0004 -w 4 \
    --save-model-to $REPO_DIR/trained/da-vinci --save-model-every 10 \
    --save-results-to $REPO_DIR/results/da-vinci --evaluate-every 10 \
    --home $BITBUCKET_HOME
