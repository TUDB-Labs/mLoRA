#!/bin/bash
# Install extra requirements
pip install ninja
pip install xformers==0.0.23.post1
pip install bitsandbytes==0.41.3
pip install flash-attn==2.5.6 --no-build-isolation
