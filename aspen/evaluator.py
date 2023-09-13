import nltk.translate.bleu_score as bleu
from rouge import Rouge as en_rouge
from rouge_chinese import Rouge as zh_rouge
import jieba


class Evaluator:
    def __init__(self):
        self.rouge_zh = zh_rouge()
        self.rouge_en = en_rouge()

    def is_contains_chinese(self, strs):
        for _char in strs:
            if '\u4e00' <= _char <= '\u9fa5':
                return True
        return False

    def calculate_ROUGE(self, generated_summary, reference_summary):
        if self.is_contains_chinese(reference_summary):
            rouger = self.rouge_zh
            generated_summary = ' '.join(jieba.cut(generated_summary))
            reference_summary = ' '.join(jieba.cut(reference_summary))
        else:
            rouger = self.rouge_en

        scores = rouger.get_scores(generated_summary, reference_summary)
        rouge_1_score = scores[0]['rouge-1']['f']
        rouge_2_score = scores[0]['rouge-2']['f']
        rouge_l_score = scores[0]['rouge-l']['f']
        return {'rouge-1': rouge_1_score, 'rouge-2': rouge_2_score, 'rouge-l': rouge_l_score}

    def calculate_BLEU(self, generated_summary, reference_summary, n):
        # Tokenize the generated summary and reference summary
        if self.is_contains_chinese(reference_summary):
            generated_summary = ' '.join(jieba.cut(generated_summary))
            reference_summary = ' '.join(jieba.cut(reference_summary))

        generated_tokens = generated_summary.split()
        reference_tokens = reference_summary.split()

        # Calculate the BLEU score
        weights = [1.0 / n] * n  # Weights for n-gram precision calculation
        bleu_score = bleu.sentence_bleu([reference_tokens], generated_tokens, weights=weights)

        return {"bleu-%s" % n: bleu_score}
