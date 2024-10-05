from typing import Tuple

from transformers import AutoTokenizer

from mlora.model.args import Masks, Tokens


class Tokenizer:
    def __init__(self, model_path: str):
        self.tokenizer_ = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.n_words_ = self.tokenizer_.vocab_size
        self.bos_id_ = self.tokenizer_.bos_token_id
        self.eos_id_ = self.tokenizer_.eos_token_id
        self.pad_id_ = self.tokenizer_.pad_token_id
        self.unk_id_ = self.tokenizer_.unk_token_id
        # maybe pad id is unk
        if self.pad_id_ is None:
            assert self.unk_id_ is not None or self.eos_id_ is not None
            self.pad_id_ = self.unk_id_ if self.unk_id_ is not None else self.eos_id_

    def encode(self, data: str, bos=True, eos=True, cutoff_len=4096) -> Tokens:
        tokens = self.tokenizer_.encode(data, add_special_tokens=False)
        tokens = tokens[: cutoff_len - int(bos) - int(eos)]
        if bos:
            tokens = [self.bos_id_] + tokens
        if eos:
            tokens = tokens + [self.eos_id_]
        return tokens

    def decode(self, data: Tokens) -> str:
        return self.tokenizer_.decode(data)

    @property
    def __padding_side(self) -> str:
        return self.tokenizer_.padding_side

    def expand_tokens(self, tokens: Tokens, align_len: int) -> Tuple[Tokens, Masks]:
        # get the mask from tokens, if True, will mask those token
        masks = [False] * len(tokens)
        if len(tokens) >= align_len:
            return tokens, masks

        pad_tokens = [self.pad_id_] * (align_len - len(tokens))
        pad_masks = [True] * (align_len - len(tokens))
        if self.__padding_side == "right":
            return tokens + pad_tokens, masks + pad_masks

        return pad_tokens + tokens, pad_masks + masks
