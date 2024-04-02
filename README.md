# m-LoRA: Efficient LLM Model Fine-Tune via Multi-LoRA Optimization
[![](https://img.shields.io/github/stars/mikecovlee/mlora?logo=GitHub)](https://github.com/mikecovlee/mlora/stargazers)
[![](https://img.shields.io/github/license/mikecovlee/mlora)](http://www.apache.org/licenses/LICENSE-2.0)
[![](https://img.shields.io/github/languages/top/mikecovlee/mlora)](https://www.python.org/)  

m-LoRA (a.k.a Multi-Lora Fine-Tune) is an open-source framework for fine-tuning Large Language Models (LLMs) using the efficient multiple LoRA/QLoRA methods. Key features of m-LoRA include:

- Efficient LoRA/QLoRA: Optimizes the fine-tuning process, significantly reducing GPU memory usage by leveraging a shared frozen-based model.

- Multiple LoRA Adapters: Support for concurrent fine-tuning of multiple LoRA/QLoRA adapters.

- LoRA-based Mix-of-Expert LLM Adapter: [MixLoRA](./MixLoRA.md), which implements a Mix-of-Expert architecture based on multiple LoRA adapters for the frozen FFN layer.

## Note from the maintainer of this repository

This is an actively developing fork of the official m-LoRA repository, focusing on LoRA + MoE and its related improvements, maintained by the authors of m-LoRA.

Please note that the functions, interfaces, and performance of this fork are slightly different from the original m-LoRA. We cannot guarantee compatibility. For production use, please prefer the [original m-LoRA](https://github.com/TUDB-Labs/multi-lora-fine-tune).

## Supported Pre-trained Models

|         | Model                                                    | # Parameters    |
|---------|----------------------------------------------------------|-----------------|
| &check; | [LLaMA](https://github.com/facebookresearch/llama)       | 7B/13B/33B/65B  |
| &check; | [LLaMA-2](https://huggingface.co/meta-llama)             | 7B/13B/70B      |
| &check; | [Qwen-2](https://qwenlm.github.io)                       | 4B/7B/14B/72B   |
| &check; | [Mistral](https://mistral.ai)                            | 7B              |

## Supported LoRA Variants

|         | LoRA Variants                                            | Arguments*                |
|---------|----------------------------------------------------------|---------------------------|
| &check; | [LoRA+](https://arxiv.org/abs/2402.12354)                | `loraplus_lr_ratio: 20.0` |
| &check; | [DoRA](https://arxiv.org/abs/2402.09353)                 | `use_dora: true`          |
| &check; | [rsLoRA](https://arxiv.org/abs/2312.03732)               | `use_rslora: true`        |

*: Arguments of configuration file

## Supported Attention Methods

|         | Attention Methods                                        | Name                  | Arguments*     |
|---------|----------------------------------------------------------|-----------------------|----------------|
| &check; | [Scaled Dot Product](https://arxiv.org/abs/1706.03762)   | `"eager"`             | `--eager_attn` |
| &check; | [xFormers](https://github.com/facebookresearch/xformers) | `"xformers"`          | None           |
| &check; | [Flash Attention 2](https://arxiv.org/abs/2307.08691)    | `"flash_attention_2"` | `--flash_attn` |

*: Arguments of `mlora.py`

If the Attention method is not specified, we automatically use xFormers during training and Scaled Dot Product during inference and evaluation. It should be noted that xFormers attention needs to align the sequence length to a multiple of 8, otherwise an error will occur.

To use flash attention 2, you need to install additional dependencies manually.

```bash
pip install ninja==1.10.2.4
pip install flash-attn==2.3.6 --no-build-isolation
```

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

It's crucial to note that regardless of the settings, **LoRA weights are always calculated and stored at full precision**. For maintaining calculation accuracy, m-LoRA framework mandates the use of full precision for calculations when accuracy is imperative.

For users with NVIDIA Ampere or newer GPU architectures, the `--tf32` option can be utilized to enable full-precision calculation acceleration.

## Quickstart

Firstly, you should clone this repository and install dependencies:
```bash
# Clone Repository
git clone https://github.com/mikecovlee/mlora
cd mlora
# Optional but recommended
conda create -n mlora python=3.10
conda activate mlora
# Install requirements
pip install -r requirements.txt
```

Secondly, you can use m-LoRA via `launch.py` conveniently:
```bash
# Grant execution permission to the file
chmod +x launch.py
# Generating configuration
launch.py gen --template lora --tasks yahma/alpaca-cleaned
# Running the training task
launch.py run --base_model yahma/llama-7b-hf
```

For further detailed usage information, please use `help` command:
```bash
launch.py help
```

## m-LoRA

The `mlora.py` code is a starting point for finetuning on various datasets.
Basic command for finetuning a baseline model on the [Alpaca Cleaned](https://github.com/gururise/AlpacaDataCleaned) dataset:
```bash
CUDA_VISIBLE_DEVICES=0 python mlora.py \
  --base_model yahma/llama-7b-hf \
  --config ./config/alpaca.json \
  --load_16bit
```

You can check the template finetune configuration in [template](./template/) folder.

For further detailed usage information, please use `--help` option:
```bash
python mlora.py --help
```

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
  howpublished = {\url{https://github.com/mikecovlee/mlora}},
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
