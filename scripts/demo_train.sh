#!/bin/bash

. venv/bin/activate

python main.py config.yml da-vinci -e 120 -b 8 -w 8 \
    --ensemble-path ensemble-dataset/da-vinci \
    --validation-size 1000 \
    --save-model-to trained/da-vinci --save-model-every 10 \
    --save-results-to results/da-vinci --evaluate-every 10 \
    --home ../