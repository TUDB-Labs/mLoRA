# m-LoRA: An Efficient LLM Fine-tuning Framework
[![](https://github.com/mikecovlee/mLoRA/actions/workflows/python-test.yml/badge.svg)](https://github.com/mikecovlee/mLoRA/actions/workflows/python-test.yml)
[![](https://img.shields.io/github/stars/mikecovlee/mLoRA?logo=GitHub&style=flat)](https://github.com/mikecovlee/mLoRA/stargazers)
[![](https://img.shields.io/github/v/release/mikecovlee/mLoRA?logo=Github)](https://github.com/mikecovlee/mLoRA/releases/latest)
[![](https://img.shields.io/pypi/v/mlora?logo=pypi)](https://pypi.org/project/mlora/)
[![](https://img.shields.io/docker/v/mikecovlee/mlora?logo=Docker&label=docker)](https://hub.docker.com/r/mikecovlee/mlora/tags)
[![](https://img.shields.io/github/license/mikecovlee/mLoRA)](http://www.apache.org/licenses/LICENSE-2.0)

m-LoRA (short for Multi-LoRA) is an open-source LLMOps framework developed by the IDs Lab at Sichuan University. It is designed for high-throughput fine-tuning, evaluation, and inference of Large Language Models (LLMs) using techniques such as LoRA, DoRA, MixLoRA, and others. Key features of mLoRA include:

- Concurrent fine-tuning of multiple adapters with a shared pre-trained model.

- Support for multiple PEFT algorithms and various pre-trained models.

- Exclusive Mo-LoRA (Mixture of LoRAs) optimization for [MixLoRA](https://github.com/TUDB-Labs/MixLoRA).

You can try m-LoRA with [Google Colab](https://githubtocolab.com/mikecovlee/mLoRA/blob/main/misc/finetune-demo.ipynb) before local installation.

## Note from the maintainer of this repository

This is an actively developing fork of the official m-LoRA repository, focusing on the PEFT algorithm and its related improvements. It is maintained by the authors of m-LoRA. Currently, this fork **does not support pipeline parallelism** and can only utilize a single compute device, such as a GPU or NPU, for each m-LoRA process. Please note that the functions, interfaces, and performance of this fork differ from those of the original m-LoRA. Compatibility is not guaranteed. For production use, please prefer the [original m-LoRA](https://github.com/TUDB-Labs/mLoRA).

## Supported Platform

| OS      | Backend | Model Precision        | Quantization  | Flash Attention |
|---------|---------|------------------------|---------------|-----------------|
| Linux   | CUDA    | FP32, FP16, TF32, BF16 | 8bit and 4bit | &check;         |
| Windows | CUDA    | FP32, FP16, TF32, BF16 | 8bit and 4bit | -               |
| macOS   | MPS     | FP32, FP16, BF16       | &cross;       | &cross;         |
| All     | CPU     | FP32, FP16, BF16       | &cross;       | &cross;         |

You can use the `MLORA_BACKEND_TYPE` environment variable to force m-LoRA to use a specific backend. For example, if you want m-LoRA to run only on CPU, you can set `MLORA_BACKEND_TYPE=CPU` before importing `mlora`.

## Supported Pre-trained Models

|         | Model                                            | Model Size |
|---------|--------------------------------------------------|------------|
| &check; | [LLaMA 1/2](https://huggingface.co/meta-llama)   | 7B/13B/70B |
| &check; | [LLaMA 3/3.1](https://huggingface.co/meta-llama) | 8B/70B     |
| &check; | [Yi 1/1.5](https://huggingface.co/01-ai)         | 6B/9B/34B  |
| &check; | [TinyLLaMA](https://huggingface.co/TinyLlama)    | 1.1B       |
| &check; | [Qwen 1.5/2](https://huggingface.co/Qwen)        | 0.5B ~ 72B |
| &check; | [Gemma](https://huggingface.co/google)           | 2B/7B      |
| &check; | [Gemma 2](https://huggingface.co/google)         | 9B/27B     |
| &check; | [Mistral](https://huggingface.co/mistralai)      | 7B         |
| &check; | [Phi 1.5/2](https://huggingface.co/microsoft)    | 2.7B       |
| &check; | [ChatGLM 1/2/3](https://huggingface.co/THUDM)    | 6B         |
| &check; | [GLM 4](https://huggingface.co/THUDM)            | 6B         |


## Supported PEFT Methods

|         | PEFT Methods                                             | Arguments*                                          |
|---------|----------------------------------------------------------|-----------------------------------------------------|
| &check; | [QLoRA](https://arxiv.org/abs/2402.12354)                | See *Quantize Methods*                              |
| &check; | [LoRA+](https://arxiv.org/abs/2402.12354)                | `loraplus_lr_ratio: 20.0`                           |
| &check; | [DoRA](https://arxiv.org/abs/2402.09353)                 | `use_dora: true`                                    |
| &check; | [rsLoRA](https://arxiv.org/abs/2312.03732)               | `use_rslora: true`                                  |
| &check; | [MixLoRA](https://arxiv.org/abs/2404.15159)              | See [MixLoRA](https://github.com/TUDB-Labs/MixLoRA) |

*: Arguments of configuration file

## Supported Attention Methods

|         | Attention Methods                                            | Name           | Arguments*               |
|---------|--------------------------------------------------------------|----------------|--------------------------|
| &check; | [Scaled Dot Product](https://arxiv.org/abs/1706.03762)       | `"eager"`      | `--attn_impl eager`      |
| &check; | [Flash Attention 2](https://arxiv.org/abs/2307.08691)        | `"flash_attn"` | `--attn_impl flash_attn` |
| &check; | [Sliding Window Attention](https://arxiv.org/abs/2004.05150) | -              | `--sliding_window`       |

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

## Offline Configuration

m-LoRA relies on **HuggingFace Hub** to download necessary models, datasets, etc. If you cannot access the Internet or need to deploy m-LoRA in an offline environment, please refer to the following guide.

1. Use `git-lfs` manually downloads models and datasets from [HuggingFace Hub](https://huggingface.co).
2. Set `--data_path` to the local path to datasets when executing `launch.py gen`.
3. Clone the [evaluate](https://github.com/huggingface/evaluate) code repository locally.
4. Set environment variable `MLORA_METRIC_PATH` to the local path to `metrics` folder of evaluate code repository.
5. Set `--base_model` to the local path to models when executing `launch.py run`.

Example of (4): `export MLORA_METRIC_PATH=/path-to-your-git-repo/evaluate/metrics`

## Known issues

 + Quantization with Qwen2 have no effect (same with transformers).
 + Applying quantization with DoRA will result in higher memory and computation cost (same with PEFT).
 + Sliding window attention with generate cache may product abnormal output.

## Installation

Please refer to [m-LoRA Install Guide](./Install.md).

## Quickstart

You can conveniently utilize m-LoRA via `launch.py`. The following example demonstrates a streamlined approach to training a dummy model with m-LoRA.

```bash
# Generating configuration
python launch.py gen --template lora --tasks ./tests/dummy_data.json

# Running the training task
python launch.py run --base_model TinyLlama/TinyLlama_v1.1

# Try with gradio web ui
python inference.py \
  --base_model TinyLlama/TinyLlama_v1.1 \
  --template alpaca \
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
# Generating configuration
python launch.py gen \
  --template lora \
  --tasks yahma/alpaca-cleaned

python mlora.py \
  --base_model meta-llama/Llama-2-7b-hf \
  --config mlora.json \
  --bf16
```

You can check the template finetune configuration in [templates](./templates/) folder.

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
Copyright © 2023-2024 IDs Lab, Sichuan University

This project is licensed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).
