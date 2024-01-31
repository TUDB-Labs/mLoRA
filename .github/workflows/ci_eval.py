import re

from subprocess import Popen, PIPE, STDOUT


def run_shell(shell):
    process = Popen(shell, stdout=PIPE, stderr=STDOUT)
    output, _ = process.communicate()
    return output


if __name__ == "__main__":
    mmlu_eval_cmd = ["python", "mlora.py",
                     "--base_model", "/data/llama-7b-hf",
                     "--evaluate", "mmlu:anatomy:5:5:2048",
                     "--evaluate_data", "cais/mmlu",
                     "--disable_lora", "--load_8bit"]

    # mmlu eval case
    output = str(run_shell(mmlu_eval_cmd))
    print(output)
    match = re.search(r'Accuracy: (\d+\.\d+)', output)
    if not match:
        exit(-1)
    if float(match.group(1)) < 0.4:
        exit(-1)

    exit(0)
