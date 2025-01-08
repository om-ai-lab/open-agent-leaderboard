from evaluation.base import BaseEvaluator
from typing import List

class PassRateEvaluator(BaseEvaluator):
    """This Evaluator can determine whether the prediction is valid or not."""

    def __init__(self) -> None:
        super().__init__()

    def score(self, predictions: List[str], references: List[str] = None) -> dict:
        """Calculate the pass rate based on predictions."""
        valid_count = sum(1 for pred in predictions if self.check_real_valid(pred))
        total_count = len(predictions) 
        
        pass_rate = (valid_count / total_count * 100) if total_count > 0 else 0
        return dict(pass_rate=pass_rate)

    def check_real_valid(self, answer: str) -> bool:
        """Check if the answer is valid by excluding responses with fail words."""
        return answer != "" and answer != None