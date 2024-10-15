# m-LoRA: Efficient Multi-LoRA Fine Tuning with Shared-Based Model
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Copyright (C) 2024 All Rights Reserved.
#
# Github:  https://github.com/TUDB-Labs/mLoRA

import optuna
import mlora.model
import mlora.utils
import mlora.executor
import mlora.config

def mock_train_model(rank, learning_rate, enabled_layers):
    # Mock model training
    score = rank * learning_rate * (1 if enabled_layers == 'last_2' else 2)
    return score

def objective(trial):
    #  the hyperparameter search space
    rank = trial.suggest_categorical('rank', [4, 8, 16])
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    enabled_layers = trial.suggest_categorical('enabled_layers', ['last_2', 'all', 'specific'])

    

    # Mock training for testing purposes
    eval_metric = mock_train_model(rank, learning_rate, enabled_layers)
    
    # Return the evaluation metric (for Optuna, we minimize this score)
    return eval_metric


if __name__ == "__main__":
    # Run hyperparameter search using Optuna
    study = optuna.create_study(direction='minimize')  # Set to 'minimize' since we want to reduce the score
    study.optimize(objective, n_trials=10)  # Run 10 trials (adjust as needed)

    # Output the best hyperparameters found
    print(f"Best hyperparameters found: {study.best_params}")
    print(f"Best evaluation metric: {study.best_value}")
    print(f"Number of trials completed: {len(study.trials)}")
    