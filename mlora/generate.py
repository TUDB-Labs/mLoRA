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
    adapter_name_: str = None
    prompts_: List[Union[str, Tuple[str, str]]] = None
    prompt_template_: str = None
    # Do not set these manually
    batch_start_idx_: int = -1
    batch_end_idx_: int = -1
    prompter_: Prompter = None

    # Set prompt_template_ to enable the prompter
    def generate_prompt(self, instruction: str, input: str = None) -> str:
        if self.prompt_template_ is None:
            if input is not None:
                logging.warn("Drop input when prompt template is not set.")
            return instruction

        if self.prompter_ is None:
            self.prompter_ = Prompter(self.prompt_template_)

        return self.prompter_.generate_prompt(instruction=instruction, input=input)

    def get_prompts(self) -> List[str]:
        prompts = []
        for prompt in self.prompts_:
            args = prompt if isinstance(prompt, Tuple) else (prompt, None)
            prompts.append(self.generate_prompt(*args))

        return prompts

    def get_response(self, output: str) -> str:
        if self.prompter_ is None:
            return output.strip()
        else:
            return self.prompter_.get_response(output)


def sample_top_p(probs, p, filter_value=float("-inf"), min_tokens_to_keep=1):
    sorted_logits, sorted_indices = torch.sort(probs, descending=False)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    sorted_indices_to_remove = cumulative_probs <= (1 - p)
    sorted_indices_to_remove[..., -min_tokens_to_keep:] = 0
    indices_to_remove = sorted_indices_to_remove.scatter(
        1, sorted_indices, sorted_indices_to_remove)
    return probs.masked_fill(indices_to_remove, filter_value)


def sample_top_k(probs, k, filter_value=float("-inf")):
    top_k = min(k, probs.size(-1))  # Safety check
    indices_to_remove = probs < torch.topk(probs, top_k)[0][..., -1, None]
    return probs.masked_fill(indices_to_remove, filter_value)


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
        packed_outputs[config.adapter_name_] = [config.get_response(
            output) for output in outputs[config.batch_start_idx_:config.batch_end_idx_]]

    return packed_outputs


@torch.inference_mode()
def generate(llm_model: LLMModel,
             tokenizer: Tokenizer,
             configs: List[GenerateConfig],
             temperature=1,
             top_p=0.9,
             top_k=50,
             do_sample=True,
             max_gen_len=128,
             device="cuda:0",
             stream_callback=None):
    sample_conditions = [temperature > 0,
                         top_p > 0 and top_p <= 1.0, top_k > 0]
    if not do_sample and any(sample_conditions):
        do_sample = True
        logging.warn("do_sample force to enabled.")

    device = torch.device(device)
    raw_prompts: List[Tokens] = []
    batch_data_config: List[LoraBatchDataConfig] = []
    for config in configs:
        tokens = [tokenizer.encode(prompt, True, False)
                  for prompt in config.get_prompts()]
        config.batch_start_idx_ = len(raw_prompts)
        config.batch_end_idx_ = config.batch_start_idx_ + len(tokens)
        batch_data_config.append(LoraBatchDataConfig(
            config.adapter_name_, config.batch_start_idx_, config.batch_end_idx_))
        raw_prompts.extend(tokens)

    batch_size = len(raw_prompts)
    min_tokens_len = min(len(t) for t in raw_prompts)
    max_tokens_len = max(len(t) for t in raw_prompts)
    assert max_tokens_len <= llm_model.max_seq_len_
    total_len = min(llm_model.max_seq_len_, max_gen_len + max_tokens_len)

    tokens = torch.full((batch_size, total_len),
                        tokenizer.pad_id_, dtype=torch.int64, device=device)
    for k, t in enumerate(raw_prompts):
        tokens[k, : len(t)] = torch.tensor(t, dtype=torch.int64, device=device)

    prev_pos = 0
    kv_cache = llm_model.prepare_kv_cache(batch_size, total_len)
    stop_reached = torch.tensor([False] * batch_size, device=device)
    input_text_mask = tokens != tokenizer.pad_id_
    for cur_pos in range(min_tokens_len, total_len):
        input_data = MultiLoraBatchData(
            lora_batch_data_config_=batch_data_config,
            batch_seq_len_=(cur_pos - prev_pos),
            batch_tokens_=tokens[:, prev_pos:cur_pos],
            kv_cache_=kv_cache,
            inference_model_=True)
        kv_cache.seq_pos = prev_pos
        logits, _ = llm_model.forward(input_data)
        probs = logits[:, -1]
        if temperature > 0:
            probs = probs / temperature

        if top_k > 0:
            probs = sample_top_k(probs, top_k)

        if top_p > 0 and top_p <= 1.0:
            probs = sample_top_p(probs, top_p)

        if do_sample:
            probs = torch.softmax(probs, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_token = torch.argmax(probs, dim=-1)

        next_token = next_token.reshape(-1)
        next_token = torch.where(
            input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
        )
        tokens[:, cur_pos] = next_token
        if stream_callback is not None:
            stream_callback(cur_pos, gen_outputs(
                configs, tokenizer, raw_prompts, tokens, max_gen_len))
        stop_reached |= (~input_text_mask[:, cur_pos]) & (
            next_token == tokenizer.eos_id_)
        prev_pos = cur_pos
        if all(stop_reached):
            break

    return gen_outputs(configs, tokenizer, raw_prompts, tokens, max_gen_len)
