import json
import os
#cimport ipdb; ipdb.set_trace()
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

# Custom logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model_config(config_path):
    """
    Load model configuration from a YAML file to initialize the model.
    Specific to our project needs for the LLM we're using.
    """
    if os.path.exists(config_path):
        logging.info(f"mLora Loading model config from {config_path}")
        with open(config_path, 'r') as config_file:
            return yaml.safe_load(config_file) 
    else:
        logging.warning(f"mLora No config found at {config_path}, using default model settings.")
        return None

def generate_task_config(
    task_idx: int,
    rank: int,
    alpha: float,
    dropout: float,
    target_modules: Dict[str, bool],
    lr: float
) -> TrainTaskConfig:
    # Constructing adapter configuration for task-specific LoRA setup.
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
    # Path to the dataset for our specific training case.
    datasets = {f"test_{task_idx}": "demo/data.json"}  

    return TrainTaskConfig(
        {
            "batch_size": 8,
            "mini_batch_size": 8,
            "num_epochs": 1,
            "cutoff_len": 1024,
            "save_step": 10000000,
            "name": f"test_{task_idx}",
            "type": "train",  # Change 'benchmark' to one of the allowed types, e.g., 'train'
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
    # Hyperparameters search range, chosen based on our experience with similar models.
    rank = int(trial.suggest_int('rank', 4, 64))
    alpha = float(trial.suggest_float('alpha', 0.1, 10.0))
    lr = float(trial.suggest_loguniform('learning_rate', 1e-5, 1e-2))
    dropout = float(trial.suggest_float('dropout', 0.0, 0.5))

    # Debugging to ensure correct types.
    logging.info(f"Rank type: {type(rank)}, Alpha type: {type(alpha)}, Dropout type: {type(dropout)}, LR type: {type(lr)}")

    # Targeting specific transformer modules in the LLM.
    target_modules = {"transformer": True}  
    logging.info(f"Running trial with rank={rank}, alpha={alpha}, lr={lr}, dropout={dropout}")

    # Generate task config for fine-tuning
    task_config = generate_task_config(
        task_idx=0, rank=rank, alpha=alpha, dropout=dropout, target_modules=target_modules, lr=lr
    )

    # Debugging task_config to verify it doesn't contain any Optuna references.
    logging.info(f"Generated task config: {task_config}")
    logging.info(f"Validating task_config type: {type(task_config)}")
    logging.info(f"Generated task_config attributes: {task_config.__dict__}")

    # Load model config and initialize training.
    model_config = load_model_config(args.config)

    tokenizer, model = mlora.model.load_model(args)
    config = mlora.config.MLoRAConfig(args.config)

    # Set up executor with task config.
    executor = mlora.executor.Executor(model, tokenizer, config)
    executor.add_task(task_config)

    # Hook to record the loss during training
    loss_dict = {}

    def record_loss(task):
        loss_dict[task.task_name()] = task.get_loss_value()

    executor.register_hook('step', record_loss)

    executor.execute()

    return loss_dict["test_0"]

if __name__ == "__main__":
    args = mlora.utils.get_cmd_args()

    # Load TinyLlama model for testing, based on our small model testing strategy.
    if args.base_model == "tinyllama":
        logging.info("Using TinyLlama_v1.1 model for testing")
        args.base_model = "/model/TinyLlama_v1.1"

    # Set up environment specifics for repeatability and metric logging.
    mlora.utils.setup_seed(args.seed)
    mlora.utils.setup_logging(args.log_level, args.log_file)
    mlora.utils.setup_cuda_check()
    mlora.utils.setup_metric_logger(args.metric_file)

    try:
        # Set up Optuna study for hyperparameter search.
        study = optuna.create_study(direction='minimize')

        # Run optimization over a set number of trials.
        study.optimize(objective, n_trials=20)

        logging.info(f"Best hyperparameters found: {study.best_params}")

    except FileNotFoundError as e:
        logging.error(f"Configuration file not found: {e}")
    except AssertionError as e:
        logging.error(f"Assertion error in model configuration or executor task addition: {e}")
    except Exception as e:
        logging.error(f"Unhandled error: {e}")
        raise
