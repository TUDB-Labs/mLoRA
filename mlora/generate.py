from mlora.modelargs import LoraConfig, LoraBatchDataConfig, MultiLoraBatchData
from mlora.tokenizer import Tokenizer, Tokens
from mlora.model import LLMModel, KVCache
from mlora.utils import Prompter

from dataclasses import dataclass
from typing import List
import torch


@dataclass
class GenerateConfig:
    batch_start_idx_: int = -1
    batch_end_idx_: int = -1
    adapter_name_: str = None
    prompt_template_: str = None
    prompter_: Prompter = None
    prompts_: List[str] = None

    def init(self, config: LoraConfig) -> "GenerateConfig":
        self.adapter_name_ = config.adapter_name_
        if self.prompt_template_ is not None:
            self.prompter_ = Prompter(self.prompt_template_)

        return self

    def generate_prompt(self, instruction: str, input: str = None):
        if self.prompter_ is None:
            if input is not None:
                raise RuntimeWarning("Input must format with prompter.")
            return instruction
        else:
            return self.prompter_(instruction=instruction, input=input)


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


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
        packed_outputs[config.adapter_name_] = outputs[config.batch_start_idx_:config.batch_end_idx_]

    return packed_outputs


@torch.inference_mode()
def generate(llm_model: LLMModel,
             tokenizer: Tokenizer,
             configs: List[GenerateConfig],
             temperature=0.6,
             top_p=0.9,
             max_gen_len=128,
             device="cuda:0",
             stream_callback=None):
    device = torch.device(device)
    raw_prompts: List[Tokens] = []
    batch_data_config: List[LoraBatchDataConfig] = []
    for config in configs:
        tokens = [tokenizer.encode(prompt, True, False)
                  for prompt in config.generate_prompt(instruction=config.prompts_)]
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
    kv_cache = KVCache()
    stop_reached = torch.tensor([False] * batch_size, device=device)
    input_text_mask = tokens != tokenizer.pad_id_
    for cur_pos in range(min_tokens_len, total_len):
        input_data = MultiLoraBatchData(
            lora_batch_data_config_=batch_data_config,
            batch_seq_len_=(cur_pos - prev_pos),
            batch_tokens_=tokens[:, prev_pos:cur_pos],
            inference_model_=True)
        kv_cache.seq_pos = prev_pos
        logits, _ = llm_model.forward(input=input_data, kv_cache=kv_cache)
        if temperature > 0:
            probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
            next_token = sample_top_p(probs, top_p)
        else:
            next_token = torch.argmax(logits[:, -1], dim=-1)

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
