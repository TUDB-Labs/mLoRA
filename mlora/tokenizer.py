from mlora.modelargs import Tokens, Masks

from transformers import AutoTokenizer
from typing import List, Union


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

    def encode(self, data: Union[str, List[str]],
               bos: bool = True, eos: bool = False) -> Tokens:
        if isinstance(data, str):
            data = [data]
        ret = []
        for dat in data:
            if bos and self.bos_id_ is not None:
                ret.append(self.bos_id_)
            ret.extend(self.tokenizer.encode(dat, add_special_tokens=False))
        if eos and self.eos_id_ is not None:
            ret.append(self.eos_id_)
        return ret

    def decode(self, data: Tokens) -> str:
        return self.tokenizer.decode(data)

    # get the mask from tokens
    #   example: tokens is [2, 3, pad, pad, 4, 5]
    #            output is [1, 1, 0,   0,   1, 1]
    def mask_from(self, tokens: Tokens) -> Masks:
        mask_tokens = [self.pad_id_]
        return [int(tok not in mask_tokens) for tok in tokens]
