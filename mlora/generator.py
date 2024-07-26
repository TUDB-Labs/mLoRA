import logging
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch

from mlora.backends import backend
from mlora.common import LLMBatchConfig, LLMModelInput, Tokens, cache_factory
from mlora.model import LLMModel
from mlora.prompter import Prompter
from mlora.tokenizer import Tokenizer


@dataclass
class GenerateData:
    adapter_name_: str = None
    prompt_index_: int = None
    prefix_length_: int = None
    raw_tokens_: Tokens = None


@dataclass
class GenerateConfig:
    adapter_name: str = None
    prompts: List[Union[str, Tuple[str, str]]] = None
    prompt_template: str = None
    # Generate Arguments
    batch_size: int = 8
    stop_token: str = None
    temperature: float = 1
    top_p: float = 0.9
    top_k: float = 50
    do_sample: bool = True
    repetition_penalty: float = 1.1
    renormalize_logits: bool = True
    # Do not set these manually
    prompter_: Prompter = None
    stop_token_: torch.Tensor = None
    data_: List[GenerateData] = None

    # Set prompt_template_ to enable the prompter
    def generate_prompt(self, instruction: str, input: str = None) -> str:
        if self.prompter_ is None:
            self.prompter_ = Prompter(self.prompt_template)

        return self.prompter_.generate_prompt(instruction=instruction, input=input)

    def get_prompts(self) -> List[str]:
        prompts = []
        for prompt in self.prompts:
            args = prompt if isinstance(prompt, Tuple) else (prompt, None)
            prompts.append(self.generate_prompt(*args))

        return prompts

    def get_response(self, output: str) -> str:
        if self.prompter_ is None:
            return output.strip()
        else:
            return self.prompter_.get_response(output)

    def reset_parameters(self):
        self.prompter_ = Prompter(self.prompt_template)
        self.stop_token_ = None
        self.data_ = []


def _logits_sample_top_p(probs, p, filter_value=float("-inf"), min_tokens_to_keep=1):
    sorted_logits, sorted_indices = torch.sort(probs, descending=False)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    sorted_indices_to_remove = cumulative_probs <= (1 - p)
    sorted_indices_to_remove[..., -min_tokens_to_keep:] = 0
    indices_to_remove = sorted_indices_to_remove.scatter(
        1, sorted_indices, sorted_indices_to_remove
    )
    return probs.masked_fill(indices_to_remove, filter_value)


def _logits_sample_top_k(probs, k, filter_value=float("-inf")):
    top_k = min(k, probs.size(-1))  # Safety check
    indices_to_remove = probs < torch.topk(probs, top_k)[0][..., -1, None]
    return probs.masked_fill(indices_to_remove, filter_value)


def _logits_repetition_penalty(prev_tokens, probs, penalty):
    score = torch.gather(probs, 1, prev_tokens)
    score = torch.where(score < 0, score * penalty, score / penalty)
    probs.scatter_(1, prev_tokens, score)
    return probs


def logits_process(
    probs: torch.Tensor,
    prev_tokens: torch.Tensor,
    temperature=1,
    top_p=0.9,
    top_k=50,
    do_sample=True,
    repetition_penalty=1.1,
    renormalize_logits=True,
):
    process_conditions = any([repetition_penalty > 0])
    sample_conditions = any([temperature > 0, top_p > 0 and top_p <= 1.0, top_k > 0])

    if not do_sample and sample_conditions:
        do_sample = True
        logging.warn("do_sample force to enabled.")

    if repetition_penalty > 0:
        probs = _logits_repetition_penalty(prev_tokens, probs, repetition_penalty)

    if process_conditions and renormalize_logits:
        probs = probs.log_softmax(-1)

    if temperature > 0:
        probs = probs / temperature

    if top_k > 0:
        probs = _logits_sample_top_k(probs, top_k)

    if top_p > 0 and top_p <= 1.0:
        probs = _logits_sample_top_p(probs, top_p)

    if sample_conditions and renormalize_logits:
        probs = probs.log_softmax(-1)

    if do_sample:
        probs = torch.softmax(probs, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
    else:
        next_token = torch.argmax(probs, dim=-1)

    return next_token.reshape(-1)


def _extract_effective_tokens(
    tokenizer: Tokenizer,
    prefix_length: int,
    tokens: Tokens,
    remove_prefix=True,
    remove_pad=True,
    remove_eos=True,
):
    if remove_prefix:
        tokens = tokens[prefix_length:]

    if remove_pad and tokenizer.pad_id_ in tokens:
        pad_idx = tokens.index(tokenizer.pad_id_)
        tokens = tokens[:pad_idx]

    if remove_eos and tokenizer.eos_id_ in tokens:
        stop_idx = tokens.index(tokenizer.eos_id_)
        tokens = tokens[:stop_idx]

    return tokens


def _gen_outputs(
    tokenizer: Tokenizer,
    config_dict: Dict[str, GenerateConfig],
    current_jobs: List[GenerateData],
    tokens: torch.Tensor,
):
    tokens = tokens.tolist()
    packed_outputs: Dict[str, List[str]] = {}
    for idx, data in enumerate(current_jobs):
        output = config_dict[data.adapter_name_].get_response(
            tokenizer.decode(
                _extract_effective_tokens(
                    tokenizer,
                    data.prefix_length_,
                    tokens[idx],
                    remove_prefix=True,
                    remove_pad=True,
                    remove_eos=True,
                )
            )
        )
        if data.adapter_name_ in packed_outputs:
            packed_outputs[data.adapter_name_].append(output)
        else:
            packed_outputs[data.adapter_name_] = [output]

    return packed_outputs


def _dispatch_task_in(
    configs: List[GenerateConfig],
    concurrent_jobs: int,
    strategy: str = "fair",
):
    assert strategy in ["fair", "fifo"], f"Unknown dispatch strategy {strategy}"
    current_jobs = []
    batch_config = []
    input_tokens = []
    max_tokens_len = 0
    min_tokens_len = sys.maxsize
    for config in configs:
        if len(batch_config) >= concurrent_jobs:
            break

        if len(config.data_) == 0:
            continue

        if strategy == "fair":
            per_task_jobs = max(concurrent_jobs // len(configs), 1)
        else:
            per_task_jobs = concurrent_jobs

        per_task_jobs = min(per_task_jobs, config.batch_size)

        batch_start_idx = len(input_tokens)
        while per_task_jobs > 0 and len(config.data_) > 0:
            per_task_jobs = per_task_jobs - 1
            data = config.data_.pop(0)
            current_jobs.append(data)
            tokens = data.raw_tokens_
            max_tokens_len = max(len(tokens), max_tokens_len)
            min_tokens_len = min(len(tokens), min_tokens_len)
            input_tokens.append(tokens)

        batch_config.append(
            LLMBatchConfig(
                adapter_name_=config.adapter_name,
                batch_start_idx_=batch_start_idx,
                batch_end_idx_=len(input_tokens),
            )
        )

    return (
        current_jobs,
        batch_config,
        input_tokens,
        max_tokens_len,
        min_tokens_len,
    )


def _dispatch_task_out(
    tokenizer: Tokenizer,
    config_dict: Dict[str, GenerateConfig],
    current_jobs: List[GenerateData],
    tokens: torch.Tensor,
    stop_reached: torch.Tensor,
):
    tokens = tokens.tolist()
    stop_reached = stop_reached.view(-1).tolist()
    packed_outputs: Dict[str, List[str]] = {}
    running_jobs: List[GenerateData] = []
    for idx, data in enumerate(current_jobs):
        if stop_reached[idx]:
            output = config_dict[data.adapter_name_].get_response(
                tokenizer.decode(
                    _extract_effective_tokens(
                        tokenizer,
                        data.prefix_length_,
                        tokens[idx],
                        remove_prefix=True,
                        remove_pad=True,
                        remove_eos=True,
                    )
                )
            )
            if data.adapter_name_ in packed_outputs:
                packed_outputs[data.adapter_name_].append(output)
            else:
                packed_outputs[data.adapter_name_] = [output]
        else:
            data.raw_tokens_ = _extract_effective_tokens(
                tokenizer,
                data.prefix_length_,
                tokens[idx],
                remove_prefix=False,
                remove_pad=True,
                remove_eos=False,
            )
            running_jobs.append(data)

    return packed_outputs, running_jobs


def _batch_generate(
    model: LLMModel,
    tokenizer: Tokenizer,
    max_gen_len: Optional[int],
    use_cache: bool,
    cache_implementation: Optional[str],
    stream_callback: Optional[Callable],
    config_dict: Dict[str, GenerateConfig],
    current_jobs: List[GenerateData],
    batch_config: List[LLMBatchConfig],
    input_tokens: List[Tokens],
    max_tokens_len: int,
    min_tokens_len: int,
):
    backend.empty_cache()
    device = torch.device(model.device_)
    batch_size = len(input_tokens)
    if max_gen_len is None:
        max_gen_len = model.config_.max_seq_len_ - max_tokens_len
    total_len = min(model.config_.max_seq_len_, max_gen_len + max_tokens_len)

    past_key_values = (
        cache_factory(
            cache_implementation=cache_implementation,
            config=model.model_.model_config(),
            max_batch_size=batch_size,
            max_cache_len=total_len,
        )
        if cache_implementation is not None
        else None
    )

    tokens = torch.full(
        (batch_size, total_len), tokenizer.pad_id_, dtype=torch.int64, device=device
    )
    for k, t in enumerate(input_tokens):
        tokens[k, : len(t)] = torch.tensor(t, dtype=torch.int64, device=device)

    prev_pos = 0
    stop_reached = torch.tensor([False] * batch_size, device=device)
    input_text_mask = tokens != tokenizer.pad_id_
    for cur_pos in range(min_tokens_len, total_len):
        input_data = LLMModelInput(
            batch_configs_=batch_config,
            batch_tokens_=tokens[:, prev_pos:cur_pos].tolist(),
            inference_mode_=True,
        )
        outputs = model.forward(input_data, past_key_values)
        for output in outputs:
            config = config_dict[output.adapter_name]
            start_idx = output.batch_start_idx_
            end_idx = output.batch_end_idx_

            next_token = logits_process(
                output.logits[:, -1],
                tokens[start_idx:end_idx, :cur_pos],
                config.temperature,
                config.top_p,
                config.top_k,
                config.do_sample,
                config.repetition_penalty,
                config.renormalize_logits,
            )

            next_token = torch.where(
                input_text_mask[start_idx:end_idx, cur_pos],
                tokens[start_idx:end_idx, cur_pos],
                next_token,
            ).to(torch.int64)
            tokens[start_idx:end_idx, cur_pos] = next_token
            stop_criteria = (~input_text_mask[start_idx:end_idx, cur_pos]) & (
                next_token == config.stop_token_
            )
            stop_reached[start_idx:end_idx] |= stop_criteria

        stop_reached |= total_len - cur_pos == 1

        if any(stop_reached):
            break

        if stream_callback is not None:
            stream_callback(
                cur_pos,
                _gen_outputs(
                    tokenizer,
                    config_dict,
                    current_jobs,
                    tokens,
                ),
            )

        if use_cache:
            prev_pos = cur_pos

    return _dispatch_task_out(
        tokenizer, config_dict, current_jobs, tokens, stop_reached
    )


@torch.inference_mode()
def generate(
    model: LLMModel,
    tokenizer: Tokenizer,
    configs: List[GenerateConfig],
    max_gen_len: Optional[int] = None,
    use_cache: bool = True,
    dispatch_strategy: str = "fair",
    concurrent_jobs: Optional[int] = None,
    cache_implementation: Optional[str] = None,
    stream_callback: Optional[Callable] = None,
):
    if concurrent_jobs is None:
        concurrent_jobs = len(configs)
        logging.info(f"Setting concurrent jobs to {concurrent_jobs} automatically")

    assert concurrent_jobs > 0

    # prepare for generation
    device = torch.device(model.device_)
    config_dict = {}
    for config in configs:
        config.reset_parameters()
        config_dict[config.adapter_name] = config
        if config.stop_token is not None:
            stop_token = tokenizer.encode(" " + config.stop_token, False)[-1]
        else:
            stop_token = tokenizer.eos_id_
        config.stop_token_ = torch.tensor(
            [stop_token], dtype=torch.int64, device=device
        )
        for idx, prompt in enumerate(config.prompts):
            args = prompt if isinstance(prompt, Tuple) else (prompt, None)
            tokens = tokenizer.encode(config.generate_prompt(*args))
            assert (
                len(tokens) < model.config_.max_seq_len_
            ), "Inputs exceeded max sequence length of model."
            config.data_.append(
                GenerateData(
                    adapter_name_=config.adapter_name,
                    prompt_index_=idx,
                    prefix_length_=len(tokens),
                    raw_tokens_=tokens,
                )
            )

    if use_cache and cache_implementation is None:
        cache_implementation = model.model_.cache_implementation()
        if cache_implementation is None:
            logging.warn(
                "Cache disabled by model, use cache_implementation to force enable."
            )
            use_cache = False

    packed_outputs: Dict[str, List] = {}

    while True:
        dispatch_args = _dispatch_task_in(configs, concurrent_jobs, dispatch_strategy)

        if len(dispatch_args[0]) == 0:
            break

        outputs, running_jobs = _batch_generate(
            model,
            tokenizer,
            max_gen_len,
            use_cache,
            cache_implementation,
            stream_callback,
            config_dict,
            *dispatch_args,
        )

        for name, output in outputs.items():
            if name in packed_outputs:
                packed_outputs[name].extend(output)
            else:
                packed_outputs[name] = output

        for data in running_jobs:
            config_dict[data.adapter_name_].data_.append(data)

    return packed_outputs
