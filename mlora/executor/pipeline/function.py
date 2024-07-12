import logging

import torch

from mlora.model.args import ModelData

from .messages import PipeMessage, PipeMessageType
from .transport import Transport


class SendOperator(torch.autograd.Function):
    # helper to reduce the activation memory
    @staticmethod
    def forward(
        ctx,
        phony: torch.Tensor,
        tensor_data: torch.Tensor,
        transport: Transport,
        msg_id: int,
        input_args: ModelData,
    ):
        assert isinstance(tensor_data, torch.Tensor)

        msg = PipeMessage(
            src_=transport.worker_name,
            dst_=transport.next_worker_name,
            msg_type_=PipeMessageType.ACTIVATIONS,
            msg_id_=msg_id,
            tensor_data_=tensor_data,
            model_data_=input_args,
            comm_data_=None,
        )
        transport.send_message(msg, False)

        return phony

    @staticmethod
    def backward(ctx, grad_output):
        assert ctx.grad_from_next_worker is not None

        return (None, ctx.grad_from_next_worker, None, None, None)


class RecvOperator(torch.autograd.Function):
    # backward will auto send the grad to pre worker
    @staticmethod
    def forward(
        ctx, phony: torch.Tensor, transport: Transport, msg: PipeMessage
    ) -> torch.Tensor:
        assert msg.msg_type_ == PipeMessageType.ACTIVATIONS
        assert isinstance(msg.tensor_data_, torch.Tensor)

        ctx.msg_id_ = msg.msg_id_
        ctx.transport_ = transport
        ctx.model_data_ = msg.model_data_

        return msg.tensor_data_ * phony

    @staticmethod
    def backward(ctx, *grad_outputs: torch.Tensor):
        transport: Transport = ctx.transport_
        if hasattr(ctx, "pre_stage_fn") and ctx.pre_stage_fn is not None:
            ctx.pre_stage_fn()

        logging.debug(f"Send the gradients to {transport.prev_worker_name}")
        transport.send_message(
            PipeMessage(
                src_=transport.worker_name,
                dst_=transport.prev_worker_name,
                msg_type_=PipeMessageType.GRADIENTS,
                msg_id_=ctx.msg_id_,
                tensor_data_=grad_outputs[0],
                model_data_=ctx.model_data_,
                comm_data_=None,
            )
        )

        return (None, None, None)
