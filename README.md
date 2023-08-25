# ASPEN: Efficient Multi-Lora Fine Tuning with Shared-Based Model

This repository provides tools for fine-tuning large language models (LLMs) using the LoRA or QLoRA methods more efficiently. It provides the framework to support multiple lora/qlora models fine tunning at the same time. By reusing the shared frozen-based model, multiple fine models can reduce GPU Memory usage greatly.  

## Table of Contents

- [Updates](#updates)
- [Overview](#overview)
- [Installation](#installation)
- [Getting Started](#started)
- [Contributing](#contributing)
- [License](#license)

## Updates

## Overview

## Installation

1. **Clone the Repository**:
   ```bash
   git clone 
   cd llm-lora-finetune
   ```

## Started
The `mlora.py` code is a starting point for finetuning and inference on various datasets.
Basic command for finetuning a baseline model on the Alpaca dataset:
```bash
python mlora.py --model_name_or_path <path_or_name>
```

For models larger than 13B, we recommend adjusting the learning rate:
```bash
python mlora.py –learning_rate 0.0001 --model_name_or_path <path_or_name>
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
  author={zhenmao ye*, dengchen li*, tingfen lan, zhaoyi liu, jie zuo, lei duan, mingjie tang},
  journal={arXiv preprint arXiv:xxxx},
  year={2023}
}
```

## License
This project is licensed under the Apache 2.0 License - see the LICENSE file for details



