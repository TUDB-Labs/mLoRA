from mlora.pipeline.messages import PipeMessage, PipeMessageType
from mlora.pipeline.transport import Transport
from mlora.model.modelargs import MultiLoraBatchData

import logging
import torch


class SendOperator(torch.autograd.Function):
    # helper to reduce the activation memory
    @staticmethod
    def forward(ctx,
                phony: torch.Tensor,
                tensor_data: torch.Tensor,
                transport: Transport,
                msg_id: int,
                input_args: MultiLoraBatchData):
        assert isinstance(tensor_data, torch.Tensor)

        msg = PipeMessage(src_=transport.worker_name,
                          dst_=transport.next_worker_name,
                          msg_type_=PipeMessageType.ACTIVATIONS,
                          msg_id_=msg_id,
                          tensor_data_=tensor_data,
                          batch_data_=input_args)
        transport.send_message(msg, False)

        return phony

    @staticmethod
    def backward(ctx, grad_output):
        assert ctx.grad_from_next_worker is not None

        return (None, ctx.grad_from_next_worker, None, None, None)


class RecvOperator(torch.autograd.Function):
    # backward will auto send the grad to pre worker
    @staticmethod
    def forward(ctx,
                phony: torch.Tensor,
                transport: Transport,
                msg: PipeMessage) -> torch.Tensor:
        assert msg.msg_type_ == PipeMessageType.ACTIVATIONS
        assert isinstance(msg.tensor_data_, torch.Tensor)

        ctx.msg_id_ = msg.msg_id_
        ctx.transport_ = transport
        ctx.batch_data_ = msg.batch_data_

        return msg.tensor_data_ * phony

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # now only signle grad can be support
        assert isinstance(grad_output, torch.Tensor)

        transport: Transport = ctx.transport_
        if hasattr(ctx, 'pre_stage_fn') and ctx.pre_stage_fn is not None:
            ctx.pre_stage_fn()

        logging.debug(f"Send the gradients to {transport.prev_worker_name}")
        transport.send_message(PipeMessage(
            src_=transport.worker_name,
            dst_=transport.prev_worker_name,
            msg_type_=PipeMessageType.GRADIENTS,
            msg_id_=ctx.msg_id_,
            tensor_data_=grad_output,
            batch_data_=ctx.batch_data_,
        ))

        return (None, None, None)
