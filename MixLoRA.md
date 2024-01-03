# MixLoRA: Resource-Efficient Model with Mix-of-Experts Architecture for Enhanced LoRA Performance

<div align="left"><img src="./assets/Mix-LoRA.png" width=60%"></div>

The fundamental concept of MixLoRA is based on a pre-trained model with all parameters frozen, such as LLaMA-7B. It involves training multiple LoRA expert modules on top of its fully connected layer (FFN). Simultaneously, a routing layer (Gate Linear) is trained, creating a more powerful Mixture of Experts (MoE) language model. Theoretically, this approach allows achieving performance similar to existing MoE models but with fewer resources.

In addition, MixLoRA also allows simultaneous fine-tuning of the attention layer, contributing to improved fine-tuning outcomes. In experiments, when the attention layer is fine-tuned simultaneously, the MixLoRA model composed of 8 experts exhibits a faster rate of loss reduction compared to the MixLoRA model with 9 experts.

MixLoRA exists within m-LoRA in a specific adapter form. Consequently, m-LoRA is capable of simultaneously loading, training, and fine-tuning multiple distinct MixLoRA models. However, it's essential to note that these models must be based on the same pre-trained model.

## Create MixLoRA model

Basic command for creating a baseline model on the [Alpaca Cleaned](https://github.com/gururise/AlpacaDataCleaned) dataset:
```bash
CUDA_VISIBLE_DEVICES=0 python mlora.py \
  --base_model yahma/llama-7b-hf \
  --config ./config/alpaca_mixlora.json \
  --load_8bit
```
Please note that once the MixLoRA model is created, the number of experts in the model cannot be changed.

## Fine-tuning MixLoRA model

The MixLoRA model can also undergo further fine-tuning.
Basic command for finetuning a model on the [Alpaca Cleaned](https://github.com/gururise/AlpacaDataCleaned) dataset:
```bash
CUDA_VISIBLE_DEVICES=0 python mlora.py \
  --base_model yahma/llama-7b-hf \
  --config ./config/alpaca_mixlora.json \
  --load_adapter \
  --load_8bit
```

## Evaluate MixLoRA model

Currently, MixLoRA supports evaluation only through the m-LoRA framework.
```bash
CUDA_VISIBLE_DEVICES=0 python mlora.py \
  --base_model yahma/llama-7b-hf \
  --config ./config/alpaca_mixlora.json \
  --load_adapter \
  --load_8bit \
  --inference
```

## Citation
Please cite the repo if you use the code in this repo.
```bibtex
@misc{Mix-LoRA,
  author = {Dengchun, Li and Tingfeng, Lan and Zhengmao, Ye and Lei, Duan and Mingjie, Tang},
  title = {MixLoRA: Resource-Efficient Model with Mix-of-Experts Architecture for Enhanced LoRA Performance},
  year = {2023},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/TUDB-Labs/multi-lora-fine-tune}},
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
