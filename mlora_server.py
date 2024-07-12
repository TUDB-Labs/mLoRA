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
import mlora.executor.task
import mlora.config
import mlora.server

import os
import plyvel
import logging
import uvicorn
import threading
import multiprocessing
from fastapi import FastAPI

m_task_done, s_task_done = multiprocessing.Pipe(True)
m_task_step, s_task_step = multiprocessing.Pipe(True)
m_task_terminate, s_task_terminate = multiprocessing.Pipe(True)


def backend_server_set_task_state(task_name: str, state: str):
    # to get the task, and set it's state
    task_info = mlora.server.db_get_obj(f"__task__{task_name}")
    if task_info is None:
        logging.info(f"the task {task_name} maybe be terminated.")
        return

    task_info["state"] = state
    mlora.server.db_put_obj(f"__task__{task_name}", task_info)

    # to get the adapter in the task, and to set it's state
    adapter_name = task_info["adapter"]
    adapter_info = mlora.server.db_get_obj(f"__adapter__{adapter_name}")
    adapter_info["state"] = state
    mlora.server.db_put_obj(f"__adapter__{adapter_name}", adapter_info)


def backend_server_delete_task(task_name: str):
    # to get the task, and set the adapters' state
    task_info = mlora.server.db_get_obj(f"__task__{task_name}")
    if task_info is None:
        logging.info(f"the task {task_name} maybe be terminated.")
        return

    # to get the adapter in the task, and to set it's state
    adapter_name = task_info["adapter"]
    adapter_info = mlora.server.db_get_obj(f"__adapter__{adapter_name}")
    adapter_info["task"] = "NO"
    mlora.server.db_put_obj(f"__adapter__{adapter_name}", adapter_info)

    mlora.server.db_del(f"__task__{task_name}")


def backend_server_run_fn(args):
    mlora.server.set_root_dir(args.root)

    root_dir_list = mlora.server.root_dir_list()
    root_dir_list = dict(
        map(lambda kv: (kv[0], os.path.join(args.root, kv[1])), root_dir_list.items())
    )

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

    web_thread = threading.Thread(
        target=uvicorn.run,
        args=(mLoRAServer,),
        kwargs={"host": "0.0.0.0", "port": 8000},
    )

    logging.info("Start the backend web server run thread")
    web_thread.start()
    logging.info("The backend web server run thread have already started")

    while True:
        if s_task_done.poll(timeout=0.1):
            task_name = s_task_done.recv()
            backend_server_set_task_state(task_name, "DONE")
        if s_task_step.poll(timeout=0.1):
            task_name, progress = s_task_step.recv()
            # the step maybe after the done
            if progress >= 100:
                continue
            backend_server_set_task_state(task_name, str(progress) + "%")
        if s_task_terminate.poll(timeout=0.1):
            task_name = s_task_terminate.recv()
            backend_server_delete_task(task_name)


def backend_model_run_fn(executor: mlora.executor.Executor):
    m_dispatcher = mlora.server.m_dispatcher()
    m_create_task = mlora.server.m_create_task()
    m_ternimate_task = mlora.server.m_notify_terminate_task()

    while True:
        if m_dispatcher.poll(timeout=0.1):
            m_dispatcher.recv()
            m_dispatcher.send(executor.dispatcher_info())
        if m_create_task.poll(timeout=0.1):
            task_conf = m_create_task.recv()
            executor.add_task(task_conf)
        if m_ternimate_task.poll(timeout=0.1):
            task_name = m_ternimate_task.recv()
            executor.notify_terminate_task(task_name)


def task_done_callback_fn(task: mlora.executor.task.Task):
    # to get the task, and set it done
    task_name = task.task_name()
    m_task_done.send(task_name)


def task_step_callback_fn(task: mlora.executor.task.Task):
    task_name = task.task_name()
    m_task_step.send((task_name, task.task_progress()))


def task_terminate_callback_fn(task: mlora.executor.task.Task):
    task_name = task.task_name()
    m_task_terminate.send(task_name)


if __name__ == "__main__":
    args = mlora.utils.get_server_cmd_args()

    mlora.utils.setup_seed(args.seed)
    mlora.utils.setup_logging(args.log_level, args.log_file)
    mlora.utils.setup_cuda_check()
    mlora.utils.setup_metric_logger(args.metric_file)

    backend_server_run_process = multiprocessing.Process(
        target=backend_server_run_fn, args=(args,)
    )
    backend_server_run_process.start()

    logging.info("Start the backend model run process")
    tokenizer, model = mlora.model.load_model(args)
    config = mlora.config.MLoRAServerConfig(
        {"name": "backend", "concurrency_num": args.concurrency_num}
    )
    if args.pipeline:
        executor = mlora.executor.PipeExecutor(
            model,
            tokenizer,
            config,
            args.device,
            args.rank,
            args.balance,
            args.recompute,
        )
    else:
        executor = mlora.executor.Executor(model, tokenizer, config)
    executor.register_hook("done", task_done_callback_fn)
    executor.register_hook("step", task_step_callback_fn)
    executor.register_hook("terminate", task_terminate_callback_fn)

    # model to execute the task
    execute_thread = threading.Thread(target=executor.execute, args=())
    logging.info("Start the backend model run thread")
    execute_thread.start()
    logging.info("The backend model run thread have already started")

    # to get command from server
    backend_model_run_fn(executor)

    execute_thread.join()
