from transformers import AutoTokenizer
from typing import List

Tokens = List[int]


class Tokenizer:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True)
        self.n_words_ = self.tokenizer.vocab_size
        self.bos_id_ = self.tokenizer.bos_token_id
        self.eos_id_ = self.tokenizer.eos_token_id
        self.pad_id_ = self.tokenizer.pad_token_id
        self.unk_id_ = self.tokenizer.unk_token_id
        # maybe pad id is unk
        if self.pad_id_ is None and self.unk_id_ is not None:
            self.pad_id_ = self.unk_id_

    def encode(self, data: str, bos: bool, eos: bool) -> Tokens:
        ret = self.tokenizer.encode(data, add_special_tokens=False)
        if bos and self.bos_id_ is not None:
            ret = [self.bos_id_] + ret
        if eos and self.eos_id_ is not None:
            ret = ret + [self.eos_id_]
        return ret

    def decode(self, data: Tokens) -> str:
        return self.tokenizer.decode(data)
