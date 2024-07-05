#!/bin/bash

black ./mlora
black ./mlora_cli
isort ./mlora --profile black
isort ./mlora_cli --profile black