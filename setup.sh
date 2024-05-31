#!/bin/bash

# setup the virtual environment
python -m venv vev
source venv/bin/activate

# upgrade pip
python -m pip install --upgrade pip

# install jax for gpu and any other requirements
python -m pip install "jax[cuda12]"
python -m pip install absl-py
