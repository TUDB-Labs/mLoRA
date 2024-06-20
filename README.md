<h1 align="center"> <p>mLoRA</p></h1>
<h3 align="center">
  <p>An Efficient "Factory" to Build Multiple LoRA Adapters</p>
</h3>

[![](https://img.shields.io/github/stars/TUDB-Labs/mLoRA?logo=GitHub)](https://github.com/TUDB-Labs/mLoRA/stargazers)
[![](https://img.shields.io/github/license/TUDB-Labs/mLoRA)](http://www.apache.org/licenses/LICENSE-2.0)
[![](https://img.shields.io/github/languages/top/TUDB-Labs/mLoRA)](https://www.python.org/)
[![](https://img.shields.io/github/v/release/TUDB-Labs/mLoRA)](https://github.com/TUDB-Labs/mLoRA/releases/latest)
[![](https://github.com/TUDB-Labs/mLoRA/actions/workflows/main-finetune-test.yaml/badge.svg)](https://github.com/TUDB-Labs/mLoRA/actions/workflows/main-finetune-test.yaml)

<div align="center">
<img src="./docs/assets/theme.webp" width="50%">
</div>


mLoRA (a.k.a Multi-LoRA Fine-Tune) is an open-source framework designed for efficient fine-tuning of multiple Large Language Models (LLMs) using LoRA and its variants. Key features of mLoRA include:

- Concurrent fine-tuning of multiple LoRA adapters.

- Shared base model among multiple LoRA adapters.

- Efficient pipeline parallelism algorithm.

- Support for multiple LoRA variant algorithms and various base models.

- Support for multiple reinforcement learning preference alignment algorithms.

## Quickstart

Firstly, you should clone this repository and install dependencies:
```bash
# Clone Repository
git clone https://github.com/TUDB-Labs/mLoRA
cd mLoRA
# Install requirements
pip install .
```

The `mlora.py` code is a starting point for batch fine-tuning LoRA adapters.
```bash
python mlora.py \
  --base_model TinyLlama/TinyLlama-1.1B-Chat-v0.4 \
  --config demo/lora/lora_case_1.yaml
```

You can check the adapters' configuration in [demo](./demo/) folder, there are some configuration regarding the use of different LoRA variants and reinforcement learning preference alignment algorithms.

For further detailed usage information, please use `--help` option:
```bash
python mlora.py --help
```

## Deploy as service
We can deploy mLoAR as a service to continuously receive user requests and perform fine-tuning task.

[![asciicast](https://asciinema.org/a/IifqdtBoJAVP4r8wg1lrcm9LI.svg)](https://asciinema.org/a/IifqdtBoJAVP4r8wg1lrcm9LI)

```bash
# Install requirements for deploy
pip install .[deploy]
# Start the server
python mlora_server.py \
  --base_model /data/TinyLlama-1.1B-Chat-v1.0/ \
  --root /tmp/mlora
```
For further detailed usage information, please use `--help` option:

```bash
python mlora_server.py --help
```

Once the service is deployed, use `mlora_cli.py` to interact with the server.

```bash
python mlora_cli.py
```

## Why you should use mLoRA

Using mLoRA can save significant computational and memory resources when training multiple adapters simultaneously.

### High performance on consumer hardware
We fine-tuned multiple LoRA adapters using four A6000 graphics cards with fp32 precision and without using checkpointing and any quantization techniques: 

| Model                                                               | mLoRA (tokens/s) | PEFT-LoRA with FSDP (tokens/s) | PEFT-LoRA with TP (tokens/s) |
| ------------------------------------------------------------------- | ---------------- | ------------------------------ | ---------------------------- |
| [llama-2-7b (32fp)](https://huggingface.co/meta-llama/llama-2-7b)   | 2364             | 1750                           | 1500                         |
| [llama-2-13b (32fp)](https://huggingface.co/meta-llama/llama-2-13b) | 1280             | OOM                            | 875                          |


### Supported model

|         | Model                            |
| ------- | -------------------------------- |
| &check; | [LLaMA](https://llama.meta.com/) |

### Supported LoRA variants

|         | Variant                                   |
| ------- | ----------------------------------------- |
| &check; | [QLoRA](https://arxiv.org/abs/2305.14314) |
| &check; | [LoRA+](https://arxiv.org/abs/2402.12354) |

### Supported preference alignment algorithms
|         | Variant                                 |
| ------- | --------------------------------------- |
| &check; | [DPO](https://arxiv.org/abs/2305.18290) |
| &check; | [CPO](https://arxiv.org/abs/2401.08417) |


## Document

- Help Document[TODO]
- [Design Document](./docs/design.md)

## Contributing
We welcome contributions to improve this repository! Please review the contribution guidelines before submitting pull requests or issues.

Fork the repository.
Create a new branch for your feature or fix.
Submit a pull request with a detailed explanation of your changes.

You can use the pre-commit to check your code.
```bash
ln -s ../../.github/workflows/pre-commit .git/hooks/pre-commit
```

## Citation
Please cite the repo if you use the code in this repo.
```bibtex
@misc{m-LoRA,
  author = {Zhengmao, Ye\textsuperscript{*} and Dengchun, Li\textsuperscript{*} and Jingqi, Tian and Tingfeng, Lan and Yanbo, Liang and Yexi, Jiang and Jie, Zuo and Hui, Lu and Lei, Duan and Mingjie, Tang},
  title = {m-LoRA: Efficient LLM Model Fine-tune and Inference via Multi-Lora Optimization},
  year = {2023},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/TUDB-Labs/mLoRA}},
  note={\textsuperscript{*}: these authors contributed equally to this work.}
}
```

## Copyright
Copyright © 2024 All Rights Reserved.

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
