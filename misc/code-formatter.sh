#!/bin/bash
black ./mlora
black ./*.py
isort ./mlora --profile black
isort ./*.py --profile black
