#!/bin/bash
black ./mlora
black ./*.py
black ./tests/*.py
isort ./mlora --profile black
isort ./*.py --profile black
isort ./tests/*.py --profile black
