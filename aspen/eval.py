import re
import nltk.translate.bleu_score as bleu
from rouge import Rouge as en_rouge
from rouge_chinese import Rouge as zh_rouge


def calculate_ROUGE(generated_summary, reference_summary):
    def is_contains_chinese(strs):
        for _char in strs:
            if '\u4e00' <= _char <= '\u9fa5':
                return True
        return False

    if is_contains_chinese(reference_summary):
        rouger = zh_rouge()
    else:
        rouger = en_rouge()

    scores = rouger.get_scores(generated_summary,reference_summary)
    rouge_1_score = scores[0]['rouge-1']['f']
    rouge_2_score = scores[0]['rouge-2']['f']
    rouge_l_score = scores[0]['rouge-l']['f']
    return {'rouge-1': rouge_1_score, 'rouge-2': rouge_2_score, 'rouge-l': rouge_l_score}

def calculate_BLEU(generated_summary, reference_summary, n):
    # n = 4, focus on the fluency of the sentence
    # Tokenize the generated summary and reference summary
    generated_tokens = generated_summary.split()
    reference_tokens = reference_summary.split()

    # Calculate the BLEU score
    weights = [1.0 / n] * n  # Weights for n-gram precision calculation
    bleu_score = bleu.sentence_bleu([reference_tokens], generated_tokens, weights=weights)

    return {"bleu-%s"%n: bleu_score}
