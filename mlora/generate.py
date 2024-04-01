from mlora.modelargs import LoraBatchDataConfig, MultiLoraBatchData
from mlora.tokenizer import Tokenizer, Tokens
from mlora.prompter import Prompter
from mlora.model import LLMModel

from typing import List, Union, Tuple
from dataclasses import dataclass
import logging
import torch


@dataclass
class GenerateConfig:
    adapter_name: str = None
    prompts: List[Union[str, Tuple[str, str]]] = None
    prompt_template: str = None
    # Generate Arguments
    temperature: float = 1
    top_p: float = 0.9
    top_k: float = 50
    do_sample: bool = True
    repetition_penalty: float = 1.1
    renormalize_logits: bool = True
    # Do not set these manually
    batch_start_idx_: int = -1
    batch_end_idx_: int = -1
    prompter_: Prompter = None

    # Set prompt_template_ to enable the prompter
    def generate_prompt(self, instruction: str, input: str = None) -> str:
        if self.prompt_template is None:
            if input is not None:
                logging.warn("Drop input when prompt template is not set.")
            return instruction

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


def _logits_sample_top_p(probs, p, filter_value=float("-inf"), min_tokens_to_keep=1):
    sorted_logits, sorted_indices = torch.sort(probs, descending=False)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    sorted_indices_to_remove = cumulative_probs <= (1 - p)
    sorted_indices_to_remove[..., -min_tokens_to_keep:] = 0
    indices_to_remove = sorted_indices_to_remove.scatter(
        1, sorted_indices, sorted_indices_to_remove)
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


def logits_process(probs: torch.Tensor,
                   prev_tokens: torch.Tensor,
                   temperature=1,
                   top_p=0.9,
                   top_k=50,
                   do_sample=True,
                   repetition_penalty=1.1,
                   renormalize_logits=True):
    process_conditions = any([repetition_penalty > 0])
    sample_conditions = any(
        [temperature > 0, top_p > 0 and top_p <= 1.0, top_k > 0])

    if not do_sample and sample_conditions:
        do_sample = True
        logging.warn("do_sample force to enabled.")

    if repetition_penalty > 0:
        probs = _logits_repetition_penalty(
            prev_tokens, probs, repetition_penalty)

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


def gen_outputs(configs, tokenizer, prompts, tokens, max_gen_len):
    outputs = []
    for i, toks in enumerate(tokens.tolist()):
        start = len(prompts[i])
        toks = toks[start: start + max_gen_len]
        if tokenizer.pad_id_ in toks:
            pad_idx = toks.index(tokenizer.pad_id_)
            toks = toks[:pad_idx]

        if tokenizer.eos_id_ in toks:
            stop_idx = toks.index(tokenizer.eos_id_)
            toks = toks[:stop_idx]

        outputs.append(tokenizer.decode(toks))

    packed_outputs = {}
    for config in configs:
        packed_outputs[config.adapter_name] = [config.get_response(
            output) for output in outputs[config.batch_start_idx_:config.batch_end_idx_]]

    return packed_outputs


@torch.inference_mode()
def generate(model: LLMModel,
             tokenizer: Tokenizer,
             configs: List[GenerateConfig],
             max_gen_len=128,
             stream_callback=None):

    device = torch.device(model.device_)
    raw_prompts: List[Tokens] = []
    batch_data_config: List[LoraBatchDataConfig] = []
    config_dict = {}
    for config in configs:
        config_dict[config.adapter_name] = config
        tokens = [tokenizer.encode(prompt, True, False)
                  for prompt in config.get_prompts()]
        config.batch_start_idx_ = len(raw_prompts)
        config.batch_end_idx_ = config.batch_start_idx_ + len(tokens)
        batch_data_config.append(LoraBatchDataConfig(
            config.adapter_name, config.batch_start_idx_, config.batch_end_idx_))
        raw_prompts.extend(tokens)

    batch_size = len(raw_prompts)
    min_tokens_len = min(len(t) for t in raw_prompts)
    max_tokens_len = max(len(t) for t in raw_prompts)
    assert max_tokens_len <= model.max_seq_len_
    total_len = min(model.max_seq_len_, max_gen_len + max_tokens_len)

    tokens = torch.full((batch_size, total_len),
                        tokenizer.pad_id_, dtype=torch.int64, device=device)
    for k, t in enumerate(raw_prompts):
        tokens[k, : len(t)] = torch.tensor(t, dtype=torch.int64, device=device)

    prev_pos = 0
    stop_reached = torch.tensor([False] * batch_size, device=device)
    input_text_mask = tokens != tokenizer.pad_id_
    for cur_pos in range(min_tokens_len, total_len):
        input_data = MultiLoraBatchData(
            lora_batch_data_config_=batch_data_config,
            batch_tokens_=tokens[:, prev_pos:cur_pos].tolist(),
            inference_seq_pos_=prev_pos)
        outputs = model.forward(input_data)
        for output in outputs:
            config = config_dict[output.adapter_name]
            start_idx = output.batch_start_idx_
            end_idx = output.batch_end_idx_

            next_token = logits_process(output.logits[:, -1],
                                        tokens[start_idx:end_idx, :cur_pos],
                                        config.temperature,
                                        config.top_p,
                                        config.top_k,
                                        config.do_sample,
                                        config.repetition_penalty,
                                        config.renormalize_logits)

            next_token = torch.where(
                input_text_mask[start_idx:end_idx,
                                cur_pos], tokens[start_idx:end_idx, cur_pos], next_token
            )
            tokens[start_idx:end_idx, cur_pos] = next_token

        if stream_callback is not None:
            stream_callback(cur_pos, gen_outputs(
                configs, tokenizer, raw_prompts, tokens, max_gen_len))

        stop_reached |= (~input_text_mask[:, cur_pos]) & (
            next_token == tokenizer.eos_id_)

        if all(stop_reached):
            break

    return gen_outputs(configs, tokenizer, raw_prompts, tokens, max_gen_len)
