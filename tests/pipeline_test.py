from pipeline_example import Pipe
from mlora.utils import setup_seed

import torch.multiprocessing as mp
import torch


def pipe_process(rank: int, world_size: int):
    setup_seed(42)
    model = Pipe(rank, world_size, device=torch.device("cuda:0"))
    model.run()
    return model.test_grads_


def test_pipe():
    world_size = 2
    ctx = mp.get_context("spawn")
    args = ((rank, world_size) for rank in range(world_size))
    with ctx.Pool(world_size) as pool:
        test_grads = pool.starmap(
            pipe_process,
            args)

    grads_1, grads_2 = test_grads

    # grads_1 and grads_2 are lists of gradients
    # compare grads_1 and grads_2 to see if they are equal
    for grad_1, grad_2 in zip(grads_1, grads_2):
        assert torch.allclose(grad_1, grad_2)

    # base_grad is the first gradient
    # test if the other gradients are multiples of the base_grad
    base_grad = grads_1[0]
    for i, grad in enumerate(grads_1):
        assert torch.allclose(base_grad * (i + 1), grad)


if __name__ == '__main__':
    test_pipe()
