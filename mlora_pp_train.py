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

import mlora.model
import mlora.utils
import mlora.executor
import mlora.config

if __name__ == "__main__":
    args = mlora.utils.get_cmd_args()

    mlora.utils.setup_seed(args.seed)
    mlora.utils.setup_logging(args.log_level, args.log_file)
    mlora.utils.setup_cuda_check()
    mlora.utils.setup_metric_logger(args.metric_file)

    # enable the trace mode for profiling performance
    if args.trace:
        mlora.utils.setup_trace_mode()

    tokenizer, model = mlora.model.load_model(args)
    config = mlora.config.MLoRAConfig(args.config)

    # init all task from config file
    executor = mlora.executor.PipeExecutor(
        model, tokenizer, config, args.device, args.rank, args.balance, args.recompute
    )

    # only the header node can add task
    if args.rank == 0:
        for item in config.tasks_:
            executor.add_task(item)

    executor.execute()
