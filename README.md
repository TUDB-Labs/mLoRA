# ASPEN: Efficient LLM Model Fine-tune and Inference via Multi-Lora Optimization
[![](https://github.com/TUDB-Labs/multi-lora-fine-tune/actions/workflows/python-test-main.yml/badge.svg)](https://github.com/TUDB-Labs/multi-lora-fine-tune/actions/workflows/python-test-main.yml)
[![](https://img.shields.io/github/stars/TUDB-Labs/multi-lora-fine-tune?logo=GitHub)](https://github.com/TUDB-Labs/multi-lora-fine-tune/stargazers)
[![](https://img.shields.io/github/license/TUDB-Labs/multi-lora-fine-tune)](http://www.apache.org/licenses/LICENSE-2.0)
[![](https://img.shields.io/github/v/release/TUDB-Labs/multi-lora-fine-tune)](https://github.com/TUDB-Labs/multi-lora-fine-tune/releases/latest)
[![](https://img.shields.io/github/languages/top/TUDB-Labs/multi-lora-fine-tune)](https://www.python.org/)  

ASPEN is an open-source framework for fine-tuning Large Language Models (LLMs) using the efficient multiple LoRA/QLoRA methods. Key features of ASPEN include:

- Efficient LoRA/QLoRA: ASPEN optimizes the fine-tuning process, significantly reducing GPU memory usage by leveraging a shared frozen-based model.

- Multiple LoRA Adapters: Support for concurrent fine-tuning of multiple LoRA/qLoRA adapters.

## Contents

- [Updates](#updates)
- [Overview](#overview)
- [Getting Started](#Quickstart)
- [Installation](#Installation)
- [Contributing](#Contributing)
- [Copyright](#Copyright)

## Updates
- Beta version: Support multiple LLaMA fine-tuning via LORA in one GPU 

## Overview

**ASPEN** is written in Python3 and compatible with HuggingFace-Transformers LLaMA Models.

This picture shows the basic principle of Multi-LoRA.

<div align="center"><img src="./assets/ASPEN-LoRA.png" width="50%"></div>

ASPEN requires [PyTorch](https://pytorch.org/) and [NVIDIA CUDA](https://developer.nvidia.com/cuda-toolkit) compatible GPUs.

### Disadvantages of LoRA-based Approaches:
- Memory Consumption: Some Lora techniques might be efficient, but the introduction of low-rank approximations can sometimes heighten memory usage, particularly if one has to store both original and approximated parameters.

- Potential for Reduced Model Accuracy: Lora-based fine-tuning is designed to either maintain or boost model accuracy. However, there can be instances where the approximations cause a dip in performance, especially if the low-rank approximations aren't chosen carefully.

- Dependence on Hyperparameters: Much like other ML techniques, Lora-based strategies involve hyperparameters that need precise fine-tuning. Mistakes in this area can lead to subpar performance.

### Improvements in LoRA-based Approaches:
- GPU Memory Conservation: Leveraging the base model more can help in significant GPU memory conservation, an essential aspect for efficient operations.
- Automatic Parameter Learning: Introducing automation in the learning process for hyperparameters during model fine-tuning can speed up the process and guarantee optimal model results.
- Early Stopping Mechanism: Implementing this approach ensures no overfitting occurs, and resources are utilized effectively. It stops training once the model's improvement becomes negligible.

### Experiment Results

Environment: NVIDIA RTX A6000 with Intel Xeon Silver 4314 on Ubuntu 22.04.3

Baseline: We utilized the widely adopted [Alpaca-LoRA](https://github.com/tloen/alpaca-lora) as a foundation. On a single GPU, we independently ran multiple Alpaca-LoRA processes in parallel (marked as *Baseline@SYNC*) and sequentially (marked as *Baseline@SEQ*), forming two baseline methods for the experiments.

#### Training Latency

<div align="center"><img src="./assets/Exp-JCT.png" width="50%"></div>

We conducted four identical fine-tuning jobs with same dataset and same hyper-parameters, incorporating two baselines and ASPEN. During the experimental process, we collected the completion times for each task in the baseline methods and calculated the time taken by the slowest task as the *Training Latency*. As shown in Figure, ASPEN exhibits lower *Training Latency* compared to both baseline methods. Specifically, ASPEN is 9.99% faster than *Baseline@SEQ* and 3.92% faster than *Baseline@SYNC*.

#### Video Memory Usage

<div align="center"><img src="./assets/Exp-Mem.png" width="75%"></div>

We conducted several fine-tuning jobs with same dataset and `batch_size = {4, 6, 8}`, incorporating  *Baseline@SYNC* and ASPEN. 

*Baseline@SYNC* triggered OOM error after 3 parallel tasks when batch size = 8, while ASPEN can handle twice that amount.

#### Batching Strategies

<div align="center"><img src="./assets/Exp-Batch.png" width="100%"></div>

We conducted four fine-tuning jobs with different dataset but same hyper-parameters, incorporating  *Baseline@SEQ* and ASPEN. 

During the experimental process, we collected following metrics:
 + *Training Latency* = Job completion time
 + *Throughput* = The number of passed tokens in model forward process / training latency
 + *Memory Usage* = Peak video memory usage
 + *GPU Utilization* = Average GPU utilization

All metrics are computed for each job. `M1, M2, M3` represent three batch strategies of ASPEN: *Optimal-Fit, Trivial, and Fast-Fit*. `BASELINE` denotes *Baseline@SEQ*.

The *Optimal-Fit* strategy performs the best across all four metrics, while the other two strategies also outperform the baseline method other than training latency.

### Use Cases:
- Domain-Specific Fine-Tuning: This involves adapting a single model with various parameters particularly for one domain.
- Cross-Domain Fine-Tuning: This method leverages the base model to fine-tune multiple models, each intended for a different domain.

## Quickstart

Firstly, you should clone this repository and install dependencies:
```bash
# Clone Repository
git clone https://github.com/TUDB-Labs/multi-lora-fine-tune
cd multi-lora-fine-tune
git submodule update --init
# Install requirements
pip install -r requirements.txt
```

The `mlora.py` code is a starting point for finetuning on various datasets.
Basic command for finetuning a baseline model on the [Alpaca Cleaned](https://github.com/gururise/AlpacaDataCleaned) dataset:
```bash
python mlora.py \
  --base_model decapoda-research/llama-7b-hf \
  --config ./config/alpaca.json \
  --load_8bit true
```

You can check the template finetune configuration in [template](./template/) folder.

For further detailed usage information, please use `--help` option:
```bash
python mlora.py --help
```
## Installation
You can also install ASPEN into your environment:
```bash
# Optional but recommended
conda create -n aspen_env python=3.8
conda activate aspen_env
# Install requirements
pip install -r requirements.txt
```
After installation, you can use ASPEN directly in your code:
```python
import aspen
```
## Contributing
We welcome contributions to improve this repository! Please review the contribution guidelines before submitting pull requests or issues.

Fork the repository.
Create a new branch for your feature or fix.
Submit a pull request with a detailed explanation of your changes.

## Citation
Please cite the repo if you use the code in this repo.
```bibtex
@misc{Multi-LoRA,
  author = {Zhengmao, Ye\textsuperscript{*} and Dengchun, Li\textsuperscript{*} and Tingfeng, Lan and Yanbo, Liang and Yexi, Jiang and Jie, Zuo and Hui, Lu and Lei, Duan and Mingjie, Tang},
  title = {ASPEN: Efficient LLM Model Fine-tune and Inference via Multi-Lora Optimization},
  year = {2023},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/TUDB-Labs/multi-lora-fine-tune}},
  note={\textsuperscript{*}: these authors contributed equally to this work.}
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
