import os
import optuna
import yaml
import logging
from typing import Dict
import mlora.config
import mlora.executor
import mlora.model
import mlora.utils
from mlora.config.adapter import LoRAConfig
from mlora.config.task import TrainTaskConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def load_model_config(config_path):
    """
    Load model configuration from a YAML file.
    """
    if os.path.exists(config_path):
        logging.info(f"Loading model config from {config_path}")
        with open(config_path, 'r') as config_file:
            return yaml.safe_load(config_file)
    else:
        logging.info(
            f"No config found at {config_path}, using default settings.")
        return None


def generate_task_config(
    task_idx: int,
    rank: int,
    alpha: float,
    dropout: float,
    target_modules: Dict[str, bool],
    lr: float
) -> TrainTaskConfig:
    # Adapter configuration for task-specific LoRA setup.
    adapters = {
        f"test_{task_idx}": LoRAConfig(
            {
                "type": "lora",
                "name": f"test_{task_idx}",
                "path": f"adapters/test_{task_idx}",
                "r": rank,
                "alpha": alpha,
                "dropout": dropout,
                "target_modules": target_modules,
                "optimizer": "adamw",
                "lr": lr,
            }
        )
    }
    datasets = {f"test_{task_idx}": "demo/data.json"}

    return TrainTaskConfig(
        {
            "batch_size": 8,
            "mini_batch_size": 8,
            "num_epochs": 1,
            "cutoff_len": 1024,
            "save_step": 10000000,
            "name": f"test_{task_idx}",
            "type": "train",
            "adapter": f"test_{task_idx}",
            "dataset": f"test_{task_idx}",
        },
        adapters,
        datasets,
    )


def objective(trial):
    """
    Objective function for Optuna hyperparameter tuning.
    """
    rank = int(trial.suggest_int('rank', 4, 64))
    alpha = float(trial.suggest_float('alpha', 0.1, 10.0))
    lr = float(trial.suggest_loguniform('learning_rate', 1e-5, 1e-2))
    dropout = float(trial.suggest_float('dropout', 0.0, 0.5))

    target_modules = {"transformer": True}
    logging.info(
        f"Running trial with rank={rank}, alpha={alpha}, lr={lr}, "
        f"dropout={dropout}"
    )

    task_config = generate_task_config(
        task_idx=0, rank=rank, alpha=alpha, dropout=dropout,
        target_modules=target_modules, lr=lr
    )

    # Load model config and initialize training
    load_model_config(args.config)

    # Load model and tokenizer
    tokenizer, model = mlora.model.load_model(args)

    # Log trainable parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            logging.info(
                f"Parameter: {name} requires_grad=True"
            )

    # Initialize other components
    config = mlora.config.MLoRAConfig(args.config)
    executor = mlora.executor.Executor(model, tokenizer, config)
    executor.add_task(task_config)

    # Record loss during training
    loss_dict = {}

    def record_loss(task):
        loss_dict[task.task_name()] = task.get_loss_value()

    executor.register_hook('step', record_loss)
    executor.execute()

    return loss_dict["test_0"]


if __name__ == "__main__":
    args = mlora.utils.get_cmd_args()

    if args.base_model == "tinyllama":
        logging.info("Using TinyLlama_v1.1 model for testing")
        args.base_model = "/model/TinyLlama_v1.1"

    # Environment setup
    mlora.utils.setup_seed(args.seed)
    mlora.utils.setup_logging(args.log_level, args.log_file)
    mlora.utils.setup_cuda_check()
    mlora.utils.setup_metric_logger(args.metric_file)

    try:
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=20)
        logging.info(
            f"Best hyperparameters found: {study.best_params}"
        )

    except FileNotFoundError as e:
        logging.error(
            f"Configuration file not found: {e}"
        )
    except AssertionError as e:
        logging.error(
            f"Assertion error in model config or exe task addition: {e}"
        )
    except Exception as e:
        logging.error(f"Unhandled error: {e}")
        raise
