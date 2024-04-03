#!/bin/bash
# Create virtual environment
conda create -n mlora python=3.10
conda activate mlora
# Install basic requirements
pip install -r requirements.txt