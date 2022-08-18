#!/bin/bash
echo "Creating virtual environment 'venv'."
python -m venv venv

# Add environment variable for falling back on CPU for any 
# operations not currently implemented on MPS.
echo "export PYTORCH_ENABLE_MPS_FALLBACK=1" >> venv/bin/activate
# Activate the environment
. ./venv/bin/activate

echo "Installing dependencies via pip."
pip install --upgrade pip
pip install -r requirements.txt

# Nightly build used for the moment since pip hasnt got the
# latest version of PyTorch with M1 GPU Acceleration.
echo "Installing nightly build of PyTorch."
pip install -U --pre torch torchvision torchaudio \
    --extra-index-url https://download.pytorch.org/whl/nightly/cpu

echo "Done."
