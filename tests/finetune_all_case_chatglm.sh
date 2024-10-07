#!/bin/bash

declare -a test_case_yamls=(
    "demo/lora/lora_case_2.yaml"
    "demo/checkpoint/checkpoint_case_3.yaml"
    "demo/checkpoint/checkpoint_case_4.yaml"
    "demo/loraplus/loraplus_case_2.yaml"
    "demo/vera/vera_case_2.yaml"
    "demo/dora/dora_case_2.yaml"
    "demo/dpo/dpo_case_4.yaml"
    "demo/dpo/dpo_case_5.yaml"
    "demo/dpo/dpo_case_6.yaml"
    "demo/cit/cit_case_2.yaml"
)

set -x
for test_case in "${test_case_yamls[@]}"
do
    python mlora_train.py --base_model $1 --config ${test_case}  --precision bf16 --model_type chatglm
done