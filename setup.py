import os

import pkg_resources
from setuptools import find_packages, setup

setup(
    name='llm-multi-lora-finetune',
    version='0.0.1',
    packages=find_packages(),
    description='A tool for fine-tuning large language models using LoRA or QLoRA',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
      str(r)
      for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
      )
    ],
    include_package_data=True,
    python_requires='>=3.6',
)