from typing import List
from collections import Counter
from parlai.core.metrics import F1Metric, AverageMetric

import re
re_art = re.compile(r'\b(a|an|the)\b')
re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')


class LenientF1Metric(F1Metric):
    """
    Helper class which computes token-level F1.
    """

    @staticmethod
    def _prec_recall_f1_score(pred_items, gold_items):
        """
        Compute precision, recall and f1 given a set of gold and prediction items.

        :param pred_items: iterable of predicted values
        :param gold_items: iterable of gold values

        :return: tuple (p, r, f1) for precision, recall, f1
        """
        # XXX: need to consider tokenizer discrepancy
        # labels are based on Spacy tokenizer, whereas model predictions are based on model-specific tokenizers
        # If predicted word contains (i.e., startswith or endswith) the label, assume that prediction is correct
        for g in gold_items:
            for idx, p in enumerate(pred_items):
                if p.startswith(g) or p.endswith(g):
                    pred_items[idx] = g

        common = Counter(gold_items) & Counter(pred_items)
        num_same = sum(common.values())
        if num_same == 0:
            return 0, 0, 0
        precision = 1.0 * num_same / len(pred_items)
        recall = 1.0 * num_same / len(gold_items)
        f1 = (2 * precision * recall) / (precision + recall)
        return precision, recall, f1

    @staticmethod
    def compute(guess: str, answers: List[str]) -> 'LenientF1Metric':
        if guess is None or answers is None:
            return AverageMetric(0, 0)
        g_tokens = normalize_answer(guess).split()
        scores = [
            LenientF1Metric._prec_recall_f1_score(g_tokens, normalize_answer(a).split())
            for a in answers
        ]
        return LenientF1Metric(max(f1 for p, r, f1 in scores), 1)

def normalize_answer(s):
    """
    Lower text and remove punctuation, articles and extra whitespace.
    """

    s = s.lower()
    s = re_punc.sub(' ', s)
    s = re_art.sub(' ', s)
    s = ' '.join(s.split())
    return s
