#!/bin/bash
# Create virtual environment
conda create -n mlora python=3.10
conda activate mlora
# Install basic requirements
pip install -r requirements.txt
# Install extra requirements
pip install xformers==0.0.24
pip install ninja==1.10.2.4
pip install flash-attn==2.3.6 --no-build-isolation