#!/bin/bash

ln -sf pyproject.cli.toml pyproject.toml
python -m build .

ln -sf pyproject.mlora.toml pyproject.toml
python -m build .