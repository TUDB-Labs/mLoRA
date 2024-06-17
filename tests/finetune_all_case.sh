#!/bin/bash

declare -a test_case_yamls=(
    "demo/dummy.yaml"
    "tests/loraplus/loraplus_case_1.yaml"
    "tests/dpo/dpo_case_1.yaml"
    "tests/dpo/dpo_case_2.yaml"
    "tests/dpo/dpo_case_3.yaml"
    "tests/cpo/cpo_case_1.yaml"
)

set -x
for test_case in "${test_case_yamls[@]}"
do
    python mlora.py --base_model $1 --config ${test_case}
done