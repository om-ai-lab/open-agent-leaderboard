from evaluation.base import BaseEvaluator
from typing import List

class MultiEvaluator:
    def __init__(self, evaluators: List[BaseEvaluator]) -> None:
        self.evaluators = evaluators

    def evaluate(self, predictions: List[str], references: List[str] = None) -> dict:
        results = {}
        for evaluator in self.evaluators:
            result = evaluator.score(predictions, references)
            results.update(result)
        return results
