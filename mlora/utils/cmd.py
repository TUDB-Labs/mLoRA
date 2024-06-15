import argparse


def get_cmd_args():
    parser = argparse.ArgumentParser(description='m-LoRA main program')
    parser.add_argument('--base_model', type=str, required=True,
                        help='Path to or name of base model')
    parser.add_argument('--model_type', type=str, default="llama",
                        help='The model type, support: llama, chatglm')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Specify which GPU to be used, default is cuda:0')
    parser.add_argument('--precision', type=str, default="int8",
                        help='Load model with different precision, include int8')
    # configuration
    parser.add_argument('--config', type=str,
                        help='Path to finetune configuration')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed in integer, default is 42')
    # configuration about log
    parser.add_argument('--log_level', type=str, default="INFO",
                        help="Set the log level.")
    parser.add_argument('--log_file', type=str,
                        help="Save log to specific file.")
    # the argument about pipeline
    parser.add_argument('--pipeline', action="store_true",
                        help="Train the LoRA model use the pipeline parallelism")
    parser.add_argument('--rank', type=int, default=-1,
                        help="The device's rank number")
    parser.add_argument('--balance', type=int, nargs="+",
                        help="The model's balance")
    # the argument about the trace mode
    parser.add_argument('--trace', action="store_true",
                        help="enbale the trace mode.")

    return parser.parse_args()
