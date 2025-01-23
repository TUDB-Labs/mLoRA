from pipeline_example import Pipe, TestModel
from mlora.utils import setup_seed

from torch.nn import Sequential
from typing import List

import torch.multiprocessing as mp
import torch


def pipe_process(rank: int, world_size: int):
    setup_seed(42)
    model = Pipe(rank, world_size, device=torch.device("cuda:0"))
    model.run()
    return model.test_grads_, model.model_.weight_, model.datas_


def test_by_pipe():
    world_size = 2

    # create 2 processes to run Pipe
    ctx = mp.get_context("spawn")
    args = ((rank, world_size) for rank in range(world_size))
    with ctx.Pool(world_size) as pool:
        res = pool.starmap(
            pipe_process,
            args)

    res_1, res_2 = res

    grads_1, weight, datas = res_1
    grads_2, _, _ = res_2

    return grads_1, grads_2, weight, datas


def test_by_sequential(weight: torch.Tensor, datas: List[torch.Tensor]):
    model_1 = TestModel(torch.device("cuda:0"))
    model_2 = TestModel(torch.device("cuda:0"))

    # set the weight of the models to the provided weight
    model_1.weight_ = weight.clone().detach()
    model_2.weight_ = weight.clone().detach()

    # enable gradient calculation for the weights
    model_1.weight_.requires_grad = True
    model_2.weight_.requires_grad = True

    grads_1 = []
    grads_2 = []

    seq_model = Sequential(model_1, model_2).cuda()

    # calculate gradients for each data in the datas list
    for data in datas:
        res = seq_model(data)
        res.sum().backward()
        grads_1.append(model_1.weight_.grad.sum())
        grads_2.append(model_2.weight_.grad.sum())

    return grads_1, grads_2


def test_pipe():
    # calculate gradients using Pipe
    grads_1, grads_2, weight, datas = test_by_pipe()

    # calculate expected gradients using Sequential with same model weight and data
    expected_grads_1, expected_grads_2 = test_by_sequential(weight, datas)

    # check if the gradients are the same
    for i in range(len(grads_1)):
        assert torch.allclose(expected_grads_1[i], grads_1[i])
        assert torch.allclose(expected_grads_2[i], grads_2[i])


if __name__ == '__main__':
    test_pipe()
