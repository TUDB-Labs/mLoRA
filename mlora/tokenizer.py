from .common import Tokens, Masks

from transformers import AutoTokenizer
from typing import List, Union


class Tokenizer:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.vocab_size_ = self.tokenizer.vocab_size
        self.bos_id_ = self.tokenizer.bos_token_id
        self.eos_id_ = self.tokenizer.eos_token_id
        self.pad_id_ = self.tokenizer.pad_token_id
        self.unk_id_ = self.tokenizer.unk_token_id
        # maybe pad id is unk
        if self.pad_id_ is None and self.unk_id_ is not None:
            self.pad_id_ = self.unk_id_

    def encode(self, data: Union[str, List[str]], add_special_tokens: bool = True) -> Tokens:
        if isinstance(data, str):
            data = [data]
        tokens = []
        ret = self.tokenizer(
            data, add_special_tokens=add_special_tokens, return_attention_mask=False).input_ids
        for toks in ret:
            tokens.extend(toks)
        return tokens

    def decode(self, data: Tokens) -> str:
        return self.tokenizer.decode(data)

    # get the mask from tokens
    #   example: tokens is [2, 3, pad, pad, 4, 5]
    #            output is [1, 1, 0,   0,   1, 1]
    def mask_from(self, tokens: Tokens) -> Masks:
        mask_tokens = [self.pad_id_]
        return [int(tok not in mask_tokens) for tok in tokens]
