#!/bin/bash

python main.py config.yml da-vinci --epochs 50 \
    --validation-size 1000 \
    --finetune-from trained/davinci/l1/final.pt \
    --save-model-to trained/da-vinci --save-model-every 10 \
    --save-results-to results/da-vinci --evaluate-every 10 \
    --home ../
