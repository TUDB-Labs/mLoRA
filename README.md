# m-LoRA: Efficient LLM Model Fine-Tune via Multi-LoRA Optimization
[![](https://github.com/scukdde-llm/mlora/actions/workflows/python-test.yml/badge.svg)](https://github.com/scukdde-llm/mlora/actions/workflows/python-test.yml)
[![](https://github.com/scukdde-llm/mlora/actions/workflows/mlora-test.yml/badge.svg)](https://github.com/scukdde-llm/mlora/actions/workflows/mlora-test.yml)
[![](https://img.shields.io/github/stars/scukdde-llm/mlora?logo=GitHub)](https://github.com/scukdde-llm/mlora/stargazers)
[![](https://img.shields.io/github/license/scukdde-llm/mlora)](http://www.apache.org/licenses/LICENSE-2.0)
[![](https://img.shields.io/github/v/release/scukdde-llm/mlora)](https://github.com/scukdde-llm/mlora/releases/latest)
[![](https://img.shields.io/github/languages/top/scukdde-llm/mlora)](https://www.python.org/)  

m-LoRA (a.k.a Multi-Lora Fine-Tune) is an open-source framework for fine-tuning Large Language Models (LLMs) using the efficient multiple LoRA/QLoRA methods. Key features of m-LoRA include:

- Efficient LoRA/QLoRA: Optimizes the fine-tuning process, significantly reducing GPU memory usage by leveraging a shared frozen-based model.

- Multiple LoRA Adapters: Support for concurrent fine-tuning of multiple LoRA/QLoRA adapters.

- LoRA-based Mix-of-Expert LLM Adapter: [MixLoRA](./docs/MixLoRA.md), which implements a Mix-of-Expert architecture based on multiple LoRA adapters for the frozen FFN layer.

## Note from the maintainer of this repository

This is an actively developing fork of the official m-LoRA repository, focusing on LoRA + MoE and its related improvements, maintained by the authors of m-LoRA.

Please note that the functions, interfaces, and performance of this fork are slightly different from the original m-LoRA. We cannot guarantee compatibility. For production use, please prefer the [original m-LoRA](https://github.com/TUDB-Labs/multi-lora-fine-tune).

## Supported Platform

| OS      | Backend | Model Precision        | Quantization  | xFormers | Flash Attention |
|---------|---------|------------------------|---------------|----------|-----------------|
| Linux   | CUDA    | FP32, FP16, TF32, BF16 | 8bit and 4bit | &check;  | &check;         |
| Windows | CUDA    | FP32, FP16, TF32, BF16 | &cross;       | &cross;  | &cross;         |
| macOS   | MPS     | FP32, FP16             | &cross;       | &cross;  | &cross;         |

**Note**: Windows and macOS support are experimental feature.

## Supported Pre-trained Models

|         | Model                                                    | # Parameters    |
|---------|----------------------------------------------------------|-----------------|
| &check; | [LLaMA](https://github.com/facebookresearch/llama)       | 7B/13B/33B/65B  |
| &check; | [LLaMA-2](https://huggingface.co/meta-llama)             | 7B/13B/70B      |
| &check; | [Qwen-2](https://qwenlm.github.io)                       | 4B/7B/14B/72B   |
| &check; | [Mistral](https://mistral.ai)                            | 7B              |

## Supported LoRA Variants

|         | LoRA Variants                                            | Arguments*                       |
|---------|----------------------------------------------------------|----------------------------------|
| &check; | [QLoRA](https://arxiv.org/abs/2402.12354)                | See *Quantize Methods*           |
| &check; | [LoRA+](https://arxiv.org/abs/2402.12354)                | `loraplus_lr_ratio: 20.0`        |
| &check; | [DoRA](https://arxiv.org/abs/2402.09353)                 | `use_dora: true`                 |
| &check; | [rsLoRA](https://arxiv.org/abs/2312.03732)               | `use_rslora: true`               |
| &check; | MixLoRA                                                  | See [MixLoRA](./docs/MixLoRA.md) |

*: Arguments of configuration file

## Supported Attention Methods

|         | Attention Methods                                        | Name           | Arguments*               |
|---------|----------------------------------------------------------|----------------|--------------------------|
| &check; | [Scaled Dot Product](https://arxiv.org/abs/1706.03762)   | `"eager"`      | `--attn_impl eager`      |
| &check; | [xFormers](https://github.com/facebookresearch/xformers) | `"xformers"`   | `--attn_impl xformers`   |
| &check; | [Flash Attention 2](https://arxiv.org/abs/2307.08691)    | `"flash_attn"` | `--attn_impl flash_attn` |

*: Arguments of `mlora.py`

m-LoRA only supports scaled-dot product attention (eager) by default. Additional requirements are necessary for xFormers and flash attention.

To utilize xFormers attention, you must manually install additional dependencies:

```bash
pip3 install xformers==0.0.23.post1
```

For flash attention, manual installation of the following dependencies is required:

```bash
pip3 install ninja
pip3 install flash-attn==2.5.6 --no-build-isolation
```

If the attention method is not specified, xFormers is automatically employed during training if available, while scaled-dot product attention is used during inference and evaluation. It's important to note that xFormers attention necessitates aligning the sequence length to a multiple of 8, otherwise, an error will occur.

## Supported Quantize Methods

|         | Quantize Methods      | Arguments*    |
|---------|-----------------------|---------------|
| &check; | Tensor Float 32       | `--tf32`      |
| &check; | Half Precision (FP16) | `--fp16`      |
| &check; | Brain Float 16        | `--bf16`      |
| &check; | 8bit Quantize         | `--load_8bit` |
| &check; | 4bit Quantize         | `--load_4bit` |

*: Arguments of `mlora.py`

m-LoRA offers support for various model accuracy and quantization methods. By default, m-LoRA utilizes full precision (Float32), but users can opt for half precision (Float16) using `--fp16` or BrainFloat16 using `--bf16`. Enabling half precision reduces the model size by half, and for further reduction, quantization methods can be employed.

Quantization can be activated using `--load_4bit` for 4-bit quantization or `--load_8bit` for 8-bit quantization. However, when only quantization is enabled, m-LoRA utilizes Float32 for calculations. To achieve memory savings during training, users can combine quantization and half-precision modes.

To enable quantization support, please manually install `bitsandbytes`:

```bash
pip3 install bitsandbytes==0.41.3
```

It's crucial to note that regardless of the settings, **LoRA weights are always calculated and stored at full precision**. For maintaining calculation accuracy, m-LoRA framework mandates the use of full precision for calculations when accuracy is imperative.

For users with NVIDIA Ampere or newer GPU architectures, the `--tf32` option can be utilized to enable full-precision calculation acceleration.

## Known issues

 + DoRA with Quantization on Qwen2
 + Half Precision with Qwen2

## Installation

Please refer to [Install Guide](./docs/Install.md).

## Quickstart

You can use m-LoRA via `launch.py` conveniently:

```bash
# Generating configuration
python launch.py gen --template lora --tasks yahma/alpaca-cleaned
# Running the training task
python launch.py run --base_model yahma/llama-7b-hf
```

For further detailed usage information, please use the `help` command:

```bash
python launch.py help
```

## m-LoRA

The `mlora.py` code is a starting point for finetuning on various datasets.
Basic command for finetuning a baseline model on the [Alpaca Cleaned](https://github.com/gururise/AlpacaDataCleaned) dataset:
```bash
python mlora.py \
  --base_model yahma/llama-7b-hf \
  --config ./config/alpaca.json \
  --load_16bit
```

You can check the template finetune configuration in [template](./template/) folder.

For further detailed usage information, please use `--help` option:
```bash
python mlora.py --help
```

## Use Docker

Firstly, ensure that you have installed Docker Engine and NVIDIA Container Toolkit correctly.

After that, you can launch the container using the following typical command:

```
docker run --gpus all -it --rm mikecovlee/mlora
```

You can check all available tags from: [mikecovlee/mlora/tags](https://hub.docker.com/r/mikecovlee/mlora/tags)

Please note that this container only provides a proper environment to run m-LoRA. The codes of m-LoRA are not included.

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
  howpublished = {\url{https://github.com/scukdde-llm/mlora}},
  note={\textsuperscript{*}: these authors contributed equally to this work.}
}
```

## Copyright
Copyright © 2023-2024 All Rights Reserved.

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
