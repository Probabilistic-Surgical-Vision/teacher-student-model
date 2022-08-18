#!/bin/bash

python main.py config.yml da-vinci --epochs 2 --adversarial \
    --training-size 16 --validation-size 16 \
    --save-model-to trained/da-vinci --save-model-every 1 \
    --save-results-to results/da-vinci --evaluate-every 1 \
    --home ../
