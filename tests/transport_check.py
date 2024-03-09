from mlora.pipeline.transport import RpcTransport
from mlora.pipeline.messages import PipeMessageType, PipeMessage

import torch
import torch.multiprocessing as mp


TEST_TIMES = 10


def send_msg(worker: RpcTransport,
             task_type: PipeMessageType,
             src: str,
             dst: str,
             loop_times: int):
    for i in range(loop_times):
        message = PipeMessage(
            src_=src,
            dst_=dst,
            msg_id_=i,
            msg_type_=task_type,
            tensor_data_=torch.ones((4096, 4096)).cuda(0) * i,
            batch_data_=None,
        )
        worker.send_message(message)


def recv_msg(worker: RpcTransport,
             task_type: PipeMessageType,
             loop_times: int):
    passed = True
    for _ in range(loop_times):
        while True:
            message = worker.recv_message(msg_type=task_type, block=False)
            if message is not None:
                break
        # check the message
        if not torch.allclose(torch.ones((4096, 4096)).cuda(0) * message.msg_id_, message.tensor_data_):
            passed = False
    return passed


def worker_0():
    worker = RpcTransport(0, 2, torch.device("cuda:0"))
    send_msg(worker, PipeMessageType.ACTIVATIONS, "worker-0", "worker-1", TEST_TIMES)
    assert recv_msg(worker, PipeMessageType.GRADIENTS, TEST_TIMES)
    worker.stop()


def worker_1():
    worker = RpcTransport(1, 2, torch.device("cuda:0"))
    assert recv_msg(worker, PipeMessageType.ACTIVATIONS, TEST_TIMES)
    send_msg(worker, PipeMessageType.GRADIENTS, "worker-1", "worker-0", TEST_TIMES)
    worker.stop()


def test_transport():
    mp.set_start_method("spawn")
    p1 = mp.Process(target=worker_0)
    p2 = mp.Process(target=worker_1)
    p1.start()
    p2.start()
    p1.join()
    p2.join()


if __name__ == "__main__":
    test_transport()
