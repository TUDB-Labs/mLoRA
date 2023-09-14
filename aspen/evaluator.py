import nltk.translate.bleu_score as bleu
from rouge import Rouge as en_rouge
from rouge_chinese import Rouge as zh_rouge
import jieba


class Evaluator:
    def __init__(self):
        self.rouge_zh = zh_rouge()
        self.rouge_en = en_rouge()

    def is_contains_chinese(self, strs) -> bool:
        for _char in strs:
            if '\u4e00' <= _char <= '\u9fa5':
                return True
        return False

    def calculate_ROUGE(self, sentence_out: str, sentence_in: str) -> dict:
        if self.is_contains_chinese(sentence_in):
            rouger = self.rouge_zh
            sentence_out = ' '.join(jieba.cut(sentence_out))
            sentence_in = ' '.join(jieba.cut(sentence_in))
        else:
            rouger = self.rouge_en

        scores = rouger.get_scores(sentence_out, sentence_in)
        rouge_1_score = round(scores[0]['rouge-1']['f'], 2)
        rouge_2_score = round(scores[0]['rouge-2']['f'], 2)
        rouge_l_score = round(scores[0]['rouge-l']['f'], 2)
        return {'rouge-1': rouge_1_score, 'rouge-2': rouge_2_score, 'rouge-l': rouge_l_score}

    def calculate_BLEU(self, sentence_out: str, sentence_in: str, n_gram: int) -> dict:
        # Tokenize the generated summary and reference summary
        if self.is_contains_chinese(sentence_in):
            sentence_out = ' '.join(jieba.cut(sentence_out))
            sentence_in = ' '.join(jieba.cut(sentence_in))

        sentence_out_tokens = sentence_out.split()
        sentence_in_tokens = sentence_in.split()

        # Calculate the BLEU score
        weights = [1.0 / n_gram] * n_gram  # Weights for n-gram precision calculation
        bleu_score = round(bleu.sentence_bleu([sentence_in_tokens], sentence_out_tokens, weights=weights), 2)

        return {"bleu-%s" % n_gram: bleu_score}
