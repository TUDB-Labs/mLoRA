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

import mlora.utils
import mlora.executor
import mlora.executor.task
import mlora.config
import mlora.server

import os
import json
import plyvel
import logging
import uvicorn
import threading
import multiprocessing
from fastapi import FastAPI

m_task_done, s_task_done = multiprocessing.Pipe(True)
m_task_step, s_task_step = multiprocessing.Pipe(True)


def backend_server_set_task_state(task_name: str, state: str):
    task_info = mlora.server.db_get_str(f'__task__{task_name}')
    task_info = json.loads(task_info)
    task_info["state"] = state
    mlora.server.db_put_str(f'__task__{task_name}', json.dumps(task_info))
    # to get the adapter in the task, and to set it done
    adapter_name = task_info["adapter"]
    adapter_info = mlora.server.db_get_str(f'__adapter__{adapter_name}')
    adapter_info = json.loads(adapter_info)
    adapter_info["state"] = state
    mlora.server.db_put_str(f'__adapter__{adapter_name}', json.dumps(adapter_info))


def backend_server_run_fn(args):
    mlora.server.set_root_dir(args.root)

    root_dir_list = mlora.server.root_dir_list()
    root_dir_list = dict(map(lambda kv: (kv[0], os.path.join(
        args.root, kv[1])), root_dir_list.items()))

    mlora.server.set_root_dir_list(root_dir_list)

    logging.info(f"Load the data from those dirs: {root_dir_list}")
    for dir_name in root_dir_list.values():
        if os.path.exists(dir_name):
            continue
        os.makedirs(dir_name)

    mlora.server.set_db(plyvel.DB(root_dir_list["db"], create_if_missing=True))

    mLoRAServer = FastAPI()
    mLoRAServer.include_router(mlora.server.dispatcher_router)
    mLoRAServer.include_router(mlora.server.file_router)
    mLoRAServer.include_router(mlora.server.dataset_router)
    mLoRAServer.include_router(mlora.server.adapter_router)
    mLoRAServer.include_router(mlora.server.task_router)

    web_thread = threading.Thread(target=uvicorn.run, args=(mLoRAServer, ))

    logging.info("Start the backend web server run thread")
    web_thread.start()
    logging.info("The backend web server run thread have already started")

    while True:
        if s_task_done.poll(timeout=0.1):
            task_name = s_task_done.recv()
            backend_server_set_task_state(task_name, "DONE")
        if s_task_step.poll(timeout=0.1):
            task_name, progress = s_task_step.recv()
            backend_server_set_task_state(task_name, str(progress) + "%")


def backend_model_run_fn(executor: mlora.executor.Executor):
    m_dispatcher = mlora.server.m_dispatcher()
    m_create_task = mlora.server.m_create_task()

    while True:
        if m_dispatcher.poll(timeout=0.1):
            m_dispatcher.recv()
            m_dispatcher.send(executor.dispatcher_info())
        if m_create_task.poll(timeout=0.1):
            task_conf = m_create_task.recv()
            executor.add_task(task_conf)


def task_done_callback_fn(task: mlora.executor.task.Task):
    # to get the task, and set it done
    task_name = task.task_name()
    m_task_done.send(task_name)


def task_step_callback_fn(task: mlora.executor.task.Task):
    task_name = task.task_name()
    m_task_step.send((task_name, task.task_progress()))


if __name__ == "__main__":
    args = mlora.utils.get_server_cmd_args()

    mlora.utils.setup_seed(args.seed)
    mlora.utils.setup_logging(args.log_level, args.log_file)
    mlora.utils.setup_cuda_check()

    backend_server_run_process = multiprocessing.Process(
        target=backend_server_run_fn, args=(args,))
    backend_server_run_process.start()

    logging.info("Start the backend model run process")
    tokenizer, model = mlora.utils.load_model(args)
    config = mlora.config.MLoRAServerConfig({
        "name": "backend",
        "concurrency_num": args.concurrency_num
    })
    executor = mlora.executor.Executor(model, tokenizer, config)
    executor.register_hook("done", task_done_callback_fn)
    executor.register_hook("step", task_step_callback_fn)

    # model to execute the task
    execute_thread = threading.Thread(target=executor.execute, args=())
    logging.info("Start the backend model run thread")
    execute_thread.start()
    logging.info("The backend model run thread have already started")

    # to get command from server
    backend_model_run_fn(executor)

    execute_thread.join()
