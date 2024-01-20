import fire
import mlora
import torch
import traceback
import gradio as gr

from queue import Queue
from threading import Thread


class Iteratorize:

    """
    Transforms a function that takes a callback
    into a lazy iterator (generator).
    """

    def __init__(self, func, kwargs={}, callback=None):
        self.mfunc = func
        self.c_callback = callback
        self.q = Queue()
        self.sentinel = object()
        self.kwargs = kwargs
        self.stop_now = False

        def _callback(seq_pos, output):
            if self.stop_now:
                raise ValueError
            self.q.put(output["m-LoRA"][0])

        def gentask():
            try:
                ret = self.mfunc(callback=_callback, **self.kwargs)
            except ValueError:
                pass
            except:
                traceback.print_exc()
                pass

            self.q.put(self.sentinel)
            if self.c_callback:
                self.c_callback(ret)

        self.thread = Thread(target=gentask)
        self.thread.start()

    def __iter__(self):
        return self

    def __next__(self):
        obj = self.q.get(True, None)
        if obj is self.sentinel:
            raise StopIteration
        else:
            return obj

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_now = True


def main(base_model: str,
         template: str = None,
         lora_weights: str = "",
         load_16bit: bool = False,
         load_8bit: bool = False,
         load_4bit: bool = False,
         device: str = "cuda:0",
         server_name: str = "0.0.0.0",
         share_gradio: bool = False):

    model = mlora.LlamaModel.from_pretrained(base_model, device=device,
                                             bits=(8 if load_8bit else (
                                                 4 if load_4bit else None)),
                                             load_dtype=torch.bfloat16 if load_16bit else torch.float32)
    tokenizer = mlora.Tokenizer(base_model)

    if lora_weights:
        model.load_adapter_weight(lora_weights, "m-LoRA")
        generation_config = model.get_generate_paramas()["m-LoRA"]
    else:
        generation_config = mlora.GenerateConfig(adapter_name_="m-LoRA")

    generation_config.prompt_template_ = template

    def evaluate(
        instruction,
        input="",
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        repetition_penalty=1.1,
        max_new_tokens=128,
        stream_output=False,
    ):
        input = input.strip()
        if len(input) == 0:
            input = None

        generation_config.prompts_ = [(instruction, input)]

        generate_params = {
            "model": model,
            "tokenizer": tokenizer,
            "configs": [generation_config],
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "max_gen_len": max_new_tokens,
            "device": device
        }

        if stream_output:
            # Stream the reply 1 token at a time.

            def generate_with_callback(callback=None, **kwargs):
                mlora.generate(stream_callback=callback, **kwargs)

            def generate_with_streaming(**kwargs):
                return Iteratorize(
                    generate_with_callback, kwargs, callback=None
                )

            with generate_with_streaming(**generate_params) as generator:
                for output in generator:
                    yield output
            return  # early return for stream_output

        # Without streaming
        output = mlora.generate(**generate_params)
        yield output["m-LoRA"][0]

    gr.Interface(
        fn=evaluate,
        inputs=[
            gr.components.Textbox(
                lines=2,
                label="Instruction",
                placeholder="Could you provide an introduction to m-LoRA?",
            ),
            gr.components.Textbox(lines=2, label="Input", placeholder="none"),
            gr.components.Slider(
                minimum=0, maximum=1, value=1, label="Temperature"
            ),
            gr.components.Slider(
                minimum=0, maximum=1, value=0.9, label="Sampling Top-P"
            ),
            gr.components.Slider(
                minimum=0, maximum=100, step=1, value=40, label="Sampling Top-K"
            ),
            gr.components.Slider(
                minimum=0, maximum=2, value=1.1, label="Repetition Penalty"
            ),
            gr.components.Slider(
                minimum=1, maximum=2000, step=1, value=128, label="Max Tokens"
            ),
            gr.components.Checkbox(
                label="Stream Output", value=True
            ),
        ],
        outputs=[
            gr.components.Textbox(
                lines=5,
                label="Output",
            )
        ],
        title="m-LoRA LLM Evaluator",
        description="Evaluate basic LLaMA model and LoRA weights",  # noqa: E501
    ).queue().launch(server_name=server_name, share=share_gradio)


if __name__ == "__main__":
    fire.Fire(main)
