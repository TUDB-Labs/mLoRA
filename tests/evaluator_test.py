from aspen.evaluator import Evaluator

evaluator = Evaluator()

generated_summary = "The dog slept on the couch."
reference_summary = "The cat sat on the mat."

rouge_score = evaluator.calculate_ROUGE(generated_summary, reference_summary)
print(rouge_score)

n = 2
bleu_score = evaluator.calculate_BLEU(generated_summary, reference_summary, n)
print(bleu_score)

generated_summary = "刚刚发声，A股这种情况十分罕见！大聪明逆市抄底330亿，一篇研报引爆全球，市场逻辑生变？"
reference_summary = """刚刚过去的这个月，美股总市值暴跌了将近6万亿美元（折合人民币超过40万亿），这背后的原因可能不仅仅是加息这么简单。
最近瑞士信贷知名分析师Zoltan Polzsar撰写了一篇极其重要的文章，详细分析了现有世界秩序的崩坏本质以及美国和西方将要采取的应对策略。
在该文中，Zoltan Polzsar直指美国通胀的本质和其长期性。同期，A股市场亦出现了大幅杀跌的情况。"""

rouge_score = evaluator.calculate_ROUGE(generated_summary, reference_summary)
print(rouge_score)

n = 2
bleu_score = evaluator.calculate_BLEU(generated_summary, reference_summary, n)
print(bleu_score)
