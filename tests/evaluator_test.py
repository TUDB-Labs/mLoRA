from mlora.evaluator import Evaluator
import unittest

evaluator = Evaluator()


class TestEvaluator(unittest.TestCase):
    def test_rouge(self):
        sentence_out = "The dog slept on the couch."
        sentence_in = "The cat sat on the mat."
        rouge_score = evaluator.calculate_ROUGE(sentence_out, sentence_in)
        self.assertEqual(0.5, rouge_score['rouge-1'])
        self.assertEqual(0.2, rouge_score['rouge-2'])
        self.assertEqual(0.5, rouge_score['rouge-l'])

        sentence_out = "刚刚发声，A股这种情况十分罕见！大聪明逆市抄底330亿，一篇研报引爆全球，市场逻辑生变？"
        sentence_in = """刚刚过去的这个月，美股总市值暴跌了将近6万亿美元（折合人民币超过40万亿），这背后的原因可能不仅仅是加息这么简单。
                        最近瑞士信贷知名分析师Zoltan Polzsar撰写了一篇极其重要的文章，详细分析了现有世界秩序的崩坏本质以及美国和西方将要采取的应对策略。
                        在该文中，Zoltan Polzsar直指美国通胀的本质和其长期性。同期，A股市场亦出现了大幅杀跌的情况。"""
        rouge_score = evaluator.calculate_ROUGE(sentence_out, sentence_in)
        self.assertEqual(0.13, rouge_score['rouge-1'])
        self.assertEqual(0.02, rouge_score['rouge-2'])
        self.assertEqual(0.1, rouge_score['rouge-l'])

    def test_bleu(self):
        sentence_out = "The dog slept on the couch."
        sentence_in = "The cat sat on the mat."

        n = 2
        bleu_score = evaluator.calculate_BLEU(sentence_out, sentence_in, n)
        self.assertEqual(0.32, bleu_score['bleu-2'])

        sentence_out = "刚刚发声，A股这种情况十分罕见！大聪明逆市抄底330亿，一篇研报引爆全球，市场逻辑生变？"
        sentence_in = """刚刚过去的这个月，美股总市值暴跌了将近6万亿美元（折合人民币超过40万亿），这背后的原因可能不仅仅是加息这么简单。
                                    最近瑞士信贷知名分析师Zoltan Polzsar撰写了一篇极其重要的文章，详细分析了现有世界秩序的崩坏本质以及美国和西方将要采取的应对策略。
                                    在该文中，Zoltan Polzsar直指美国通胀的本质和其长期性。同期，A股市场亦出现了大幅杀跌的情况。"""
        bleu_score = evaluator.calculate_BLEU(sentence_out, sentence_in, n)
        self.assertEqual(0.01, bleu_score['bleu-2'])
