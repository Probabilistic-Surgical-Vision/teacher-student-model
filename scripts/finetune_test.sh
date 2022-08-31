#!/bin/bash

. venv/bin/activate

python main.py config.yml scared --e 200 -b 8 -w 8 \
    --validation-size 1000 \
    --ensemble-path PATH_TO_ENSEMBLE_DATASET \
    --finetune-from PATH_TO_MODEL_TO_FINETUNE \
    --save-model-to trained/scared --save-model-every 10 \
    --save-results-to results/scared --evaluate-every 10 \
    --home ../
