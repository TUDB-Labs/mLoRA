import fire
import mlora
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


def main(
    load_8bit: bool = False,
    base_model: str = "",
    lora_weights: str = "",
    target_device: str = "cuda:0",
    prompt_template: str = None,
    server_name: str = "0.0.0.0",
    share_gradio: bool = False,
):
    assert (
        base_model
    ), "Please specify a --base_model"

    model = mlora.LlamaModel.from_pretrained(
        base_model, device=target_device, bits=8 if load_8bit else None)
    tokenizer = mlora.Tokenizer(base_model, device=target_device)

    if lora_weights:
        model.load_adapter_weight(lora_weights, "m-LoRA")
        generation_config = model.get_generate_paramas()["m-LoRA"]
    else:
        generation_config = mlora.GenerateConfig(adapter_name_="m-LoRA")

    generation_config.prompt_template_ = prompt_template

    def evaluate(
        instruction,
        input="",
        temperature=0.1,
        top_p=0.75,
        max_new_tokens=128,
        stream_output=False,
    ):
        input = input.strip()
        if len(input) == 0:
            input = None

        generation_config.prompts_ = [
            generation_config.generate_prompt(instruction, input)]

        generate_params = {
            "llm_model": model,
            "tokenizer": tokenizer,
            "configs": [generation_config],
            "temperature": temperature,
            "top_p": top_p,
            "max_gen_len": max_new_tokens
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
                minimum=0, maximum=1, value=0.2, label="Temperature"
            ),
            gr.components.Slider(
                minimum=0, maximum=1, value=0.9, label="Top-p"
            ),
            gr.components.Slider(
                minimum=1, maximum=2000, step=1, value=128, label="Max tokens"
            ),
            gr.components.Checkbox(
                label="Stream output", value=True
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
