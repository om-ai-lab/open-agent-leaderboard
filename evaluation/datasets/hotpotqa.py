from evaluation.base import BaseEvaluator
import re
import string
from collections import Counter

# https://raw.githubusercontent.com/hotpotqa/hotpot/master/hotpot_evaluate_v1.py

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


class HotpotQaEvaluator(BaseEvaluator):
    def is_equal(self, pred, refer):
        try:
            if pred == refer or abs(float(pred) - int(refer)) < 1e-6:
                return True
        except Exception:
            pass
        return False

    def score(self, predictions, references):
        """Calculate accuracy."""
        if len(predictions) != len(references):
            return {"error": "preds and refrs have different length"}

        details = []
        total_em = 0
        total_f1 = 0
        
        for pred, refer in zip(predictions, references):
            if pred is None or refer is None:
                details.append({
                    'pred': pred,
                    'answer': refer,
                    'correct': False
                })
                continue

            normalized_pred = normalize_answer(pred)
            normalized_refer = normalize_answer(refer)
            
            em = int(normalized_pred == normalized_refer)
            f1 = f1_score(normalized_pred, normalized_refer)[0]
            
            total_em += em
            total_f1 += f1
            
            details.append({
                'pred': normalized_pred,
                'answer': normalized_refer,
                'correct': self.is_equal(pred, refer)
            })
        
        num_samples = len(predictions)
        return {
            'reward': 100 *  total_em / num_samples,
            'em': 100 *  total_em / num_samples,
            'f1': 100 *  total_f1 / num_samples,
            'details': details
        }


if __name__ == "__main__":
    import os
    import json

    evaluator = HotpotQaEvaluator()
    # Example usage

    outputs_path = "/path/to/your/local"
    with open(
        os.path.join(outputs_path, "hotpotqa.json"), "r", encoding="utf-8"
    ) as file:
        data = json.load(file)["model_result"]

        predictions = [
            item["last_output"].split("The answer is")[-1].strip() for item in data
        ]
        references = [item["ground_truth"] for item in data]

    result = evaluator.score(predictions, references)
    print(result)
