# ASPEN: Efficient Multi-LoRA Fine Tuning with Shared-Based Model
[![Python application](https://github.com/TUDB-Labs/multi-lora-fine-tune/actions/workflows/python-app.yml/badge.svg)](https://github.com/TUDB-Labs/multi-lora-fine-tune/actions/workflows/python-app.yml)

This repository provides tools for fine-tuning large language models (LLMs) using the LoRA or QLoRA methods more efficiently. It provides the framework to support multiple LoRA/qLoRA models fine tunning at the same time. By reusing the shared frozen-based model, we provide the framework to reduce GPU Memory usage for multiple fine-tuning models.  

## Table of Contents

- [Updates](#updates)
- [Overview](#overview)
- [Installation](#Installation)
- [Getting Started](#Quickstart)
- [Contributing](#Contributing)
- [Copyright](#Copyright)

## Updates
Support 

## Overview

## Installation
```bash
# Optional but recommended
conda create -n aspen_env python=3.6
# Install requirements
pip install -r requirements.txt
```
After installation, you can use ASPEN directly in your code:
```python
import aspen
```
## Quickstart

The `mlora.py` code is a starting point for finetuning and inference on various datasets.
Basic command for finetuning a baseline model on the Alpaca dataset:
```bash
python mlora.py --model_name_or_path <path_or_name>
```

For models larger than 13B, we recommend adjusting the learning rate:
```bash
python mlora.py -–learning_rate 0.0001 --model_name_or_path <path_or_name>
```

You can check detailed usage information by `--help` option:
```bash
python mlora.py --help
```
   
## Contributing
We welcome contributions to improve this repository! Please review the contribution guidelines before submitting pull requests or issues.

Fork the repository.
Create a new branch for your feature or fix.
Submit a pull request with a detailed explanation of your changes.

## Citation

```bibtex
@article{multi-lora,
  title={Aspen: Efficient Finetuning of Multiple Lora and QLora},
  author={zhengmao ye*, dengchun li*, tingfeng lan, zhaoyi liu, jie zuo, lei duan, mingjie tang},
  journal={arXiv preprint arXiv:xxxx},
  year={2023}
}
```

## Copyright
Copyright © 2023 All Rights Reserved.

This project is licensed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).

```
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```