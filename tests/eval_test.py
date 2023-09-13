from aspen import eval as aspen_eval

generated_summary = "The dog slept on the couch."
reference_summary = "The cat sat on the mat."

rouge_score = aspen_eval.calculate_ROUGE(generated_summary, reference_summary)
print(rouge_score)

n = 2
bleu_score = aspen_eval.calculate_BLEU(generated_summary, reference_summary, n)
print(bleu_score)
