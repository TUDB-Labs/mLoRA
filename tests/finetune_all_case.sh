#!/bin/bash

declare -a test_case_yamls=(
    "demo/lora/lora_case_1.yaml"
    "demo/checkpoint/checkpoint_case_1.yaml"
    "demo/checkpoint/checkpoint_case_2.yaml"
    "demo/loraplus/loraplus_case_1.yaml"
    "demo/vera/vera_case_1.yaml"
    "demo/dora/dora_case_1.yaml"
    "demo/dpo/dpo_case_1.yaml"
    "demo/dpo/dpo_case_2.yaml"
    "demo/dpo/dpo_case_3.yaml"
    "demo/cit/cit_case_1.yaml"
    "demo/ppo/ppo_case1.yaml"
)

set -x
for test_case in "${test_case_yamls[@]}"
do
    python mlora_train.py --base_model $1 --config ${test_case}
done