#!/bin/bash
# Install basic requirements
pip install -r requirements.txt
# Install extra requirements
pip install xformers==0.0.24
pip install ninja
pip install flash-attn==2.5.6 --no-build-isolation
