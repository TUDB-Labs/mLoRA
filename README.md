# m-LoRA: An Efficient "Factory" to Build Multiple LoRA Adapters
[![](https://github.com/scukdde-llm/mlora/actions/workflows/python-test.yml/badge.svg)](https://github.com/scukdde-llm/mlora/actions/workflows/python-test.yml)
[![](https://img.shields.io/github/stars/scukdde-llm/mlora?logo=GitHub&style=flat)](https://github.com/scukdde-llm/mlora/stargazers)
[![](https://img.shields.io/github/v/release/scukdde-llm/mlora?logo=Github)](https://github.com/scukdde-llm/mlora/releases/latest)
[![](https://img.shields.io/pypi/v/mlora?logo=pypi)](https://pypi.org/project/mlora/)
[![](https://img.shields.io/docker/v/mikecovlee/mlora?logo=Docker&label=docker)](https://hub.docker.com/r/mikecovlee/mlora/tags)
[![](https://img.shields.io/github/license/scukdde-llm/mlora)](http://www.apache.org/licenses/LICENSE-2.0)

mLoRA (a.k.a Multi-LoRA Fine-Tune) is an open-source framework designed for efficient fine-tuning of multiple Large Language Models (LLMs) using LoRA and its variants. Key features of mLoRA include:

- Concurrent fine-tuning of multiple LoRA adapters.

- Shared base model among multiple LoRA adapters.

- Support for multiple LoRA variant algorithms and various base models.

- Exclusive Mo-LoRA (Mixture of LoRAs) optimization for MixLoRA and its variants.

You can try m-LoRA with [Google Colab](https://githubtocolab.com/scukdde-llm/mLoRA/blob/main/mlora.ipynb) before local installation.

## Note from the maintainer of this repository

This is an actively developing fork of the official m-LoRA repository, focusing on PEFT algorithm and its related improvements, maintained by the authors of m-LoRA. Please note that the functions, interfaces, and performance of this fork are slightly different from the original m-LoRA. We cannot guarantee compatibility. For production use, please prefer the [original m-LoRA](https://github.com/TUDB-Labs/mLoRA).

## Supported Platform

| OS      | Backend | Model Precision        | Quantization  | Flash Attention |
|---------|---------|------------------------|---------------|-----------------|
| Linux   | CUDA    | FP32, FP16, TF32, BF16 | 8bit and 4bit | &check;         |
| Windows | CUDA    | FP32, FP16, TF32, BF16 | 8bit and 4bit | -               |
| macOS   | MPS     | FP32, FP16, BF16       | &cross;       | &cross;         |
| All     | CPU     | FP32, FP16, BF16       | &cross;       | &cross;         |

You can use the `MLORA_BACKEND_TYPE` environment variable to force m-LoRA to use a specific backend. For example, if you want m-LoRA to run only on CPU, you can set `MLORA_BACKEND_TYPE=CPU` before importing `mlora`.

## Supported Pre-trained Models

|         | Model                                            | # Parameters       |
|---------|--------------------------------------------------|--------------------|
| &check; | [LLaMA 1/2/3](https://huggingface.co/meta-llama) | 7B/8B/13B/70B      |
| &check; | [TinyLLaMA](https://huggingface.co/TinyLlama)    | 1.1B               |
| &check; | [Qwen 1.5/2](https://qwenlm.github.io)           | 1.5B/4B/7B/57B/72B |
| &check; | [Gemma](https://ai.google.dev/gemma/docs)        | 2B/7B              |
| &check; | [Mistral](https://mistral.ai)                    | 7B                 |
| &check; | [Phi 2](https://huggingface.co/microsoft/phi-2)  | 2.7B               |
| &check; | [ChatGLM 1/2/3/4](https://huggingface.co/THUDM)  | 6B                 |


## Supported LoRA Variants

|         | LoRA Variants                                            | Arguments*                                          |
|---------|----------------------------------------------------------|-----------------------------------------------------|
| &check; | [QLoRA](https://arxiv.org/abs/2402.12354)                | See *Quantize Methods*                              |
| &check; | [LoRA+](https://arxiv.org/abs/2402.12354)                | `loraplus_lr_ratio: 20.0`                           |
| &check; | [DoRA](https://arxiv.org/abs/2402.09353)                 | `use_dora: true`                                    |
| &check; | [rsLoRA](https://arxiv.org/abs/2312.03732)               | `use_rslora: true`                                  |
| &check; | [MixLoRA](https://arxiv.org/abs/2404.15159)              | See [MixLoRA](https://github.com/TUDB-Labs/MixLoRA) |

*: Arguments of configuration file

## Supported Attention Methods

|         | Attention Methods                                        | Name           | Arguments*               |
|---------|----------------------------------------------------------|----------------|--------------------------|
| &check; | [Scaled Dot Product](https://arxiv.org/abs/1706.03762)   | `"eager"`      | `--attn_impl eager`      |
| &check; | [Flash Attention 2](https://arxiv.org/abs/2307.08691)    | `"flash_attn"` | `--attn_impl flash_attn` |

*: Arguments of `mlora.py`

m-LoRA only supports scaled-dot product attention (eager) by default. Additional requirements are necessary for flash attention.

For flash attention, manual installation of the following dependencies is required:

```bash
pip3 install ninja
pip3 install flash-attn==2.5.8 --no-build-isolation
```

If any attention method is not specified, flash attention is used if available.

## Supported Quantize Methods

|         | Quantize Methods      | Arguments*    |
|---------|-----------------------|---------------|
| &check; | Full Precision (FP32) | by default    |
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
pip3 install bitsandbytes==0.43.1
```

It's crucial to note that regardless of the settings, **LoRA weights are always calculated and stored at full precision**. For maintaining calculation accuracy, m-LoRA framework mandates the use of full precision for calculations when accuracy is imperative.

For users with NVIDIA Ampere or newer GPU architectures, the `--tf32` option can be utilized to enable full-precision calculation acceleration.

## Known issues

 + Quantization with Qwen2 have no effect (same with transformers).
 + Applying quantization with DoRA will result in higher memory and computation cost (same with PEFT).

## Installation

Please refer to [m-LoRA Install Guide](./Install.md).

## Quickstart

You can conveniently utilize m-LoRA via `launch.py`. The following example demonstrates a streamlined approach to training a dummy model with m-LoRA.

```bash
# Generating configuration
python launch.py gen --template lora --tasks ./data/dummy_data.json
# Running the training task
python launch.py run --base_model TinyLlama/TinyLlama_v1.1
# Try with gradio web ui
python inference.py \
  --base_model TinyLlama/TinyLlama_v1.1 \
  --template ./template/alpaca.json \
  --lora_weights ./casual_0
```

For further detailed usage information, please refer to the `help` command:

```bash
python launch.py help
```

## m-LoRA

The `mlora.py` code is a starting point for finetuning on various datasets.
Basic command for finetuning a baseline model on the [Alpaca Cleaned](https://github.com/gururise/AlpacaDataCleaned) dataset:
```bash
python mlora.py \
  --base_model meta-llama/Llama-2-7b-hf \
  --config ./config/alpaca.json \
  --bf16
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

## Copyright
Copyright © 2023-2024 KDDE Lab, Sichuan University

This project is licensed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).
