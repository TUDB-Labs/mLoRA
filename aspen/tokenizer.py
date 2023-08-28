from aspen.modelargs import TokenizerArgs

from sentencepiece import SentencePieceProcessor
from typing import List


class Tokenizer:
    def __init__(self, model_path: str):
        self.token_model_ = SentencePieceProcessor(model_file=model_path)
        self.n_words_ = self.token_model_.vocab_size()
        self.bos_id_ = self.token_model_.bos_id()
        self.eos_id_ = self.token_model_.eos_id()
        self.pad_id_ = self.token_model_.pad_id()

    def encode(self, data: str, bos: bool, eos: bool) -> List[int]:
        ret = self.token_model_.encode(data)
        if bos:
            ret = [self.bos_id_] + ret
        if eos:
            ret = ret + [self.eos_id_]
        return ret

    def decode(self, data: List[int]) -> str:
        return self.token_model_.decode(data)

    def get_args(self) -> TokenizerArgs:
        return TokenizerArgs(vocab_size_=self.n_words_,
                             bos_id_=self.bos_id_,
                             eos_id_=self.eos_id_,
                             pad_id_=self.pad_id_)
