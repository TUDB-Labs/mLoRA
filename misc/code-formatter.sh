#!/bin/bash
black .
isort . --profile black
flake8 . --show-source --statistics --max-line-length=128 --max-complexity 15 --ignore=E203,W503,E722