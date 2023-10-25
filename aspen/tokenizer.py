from aspen.modelargs import TokenizerArgs

from sentencepiece import SentencePieceProcessor
from transformers import LlamaTokenizer
from typing import List

Tokens = List[int]


class Tokenizer:
    def __init__(self, model_path: str, from_file: bool = False):
        if from_file:
            self.token_model_ = SentencePieceProcessor(model_file=model_path)
        else:
            self.token_model_ = LlamaTokenizer.from_pretrained(model_path).sp_model
        self.n_words_ = self.token_model_.vocab_size()
        self.bos_id_ = self.token_model_.bos_id()
        self.eos_id_ = self.token_model_.eos_id()
        self.pad_id_ = self.token_model_.pad_id()

    def encode(self, data: str, bos: bool, eos: bool) -> Tokens:
        ret = self.token_model_.encode(data)
        if bos:
            ret = [self.bos_id_] + ret
        if eos:
            ret = ret + [self.eos_id_]
        return ret

    def decode(self, data: Tokens) -> str:
        return self.token_model_.decode(data)

    def get_args(self) -> TokenizerArgs:
        return TokenizerArgs(vocab_size_=self.n_words_,
                             bos_id_=self.bos_id_,
                             eos_id_=self.eos_id_,
                             pad_id_=self.pad_id_)
