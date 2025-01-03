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


def EM(answer, key) -> bool:
    return normalize_answer(answer) == normalize_answer(key)


def update_sp(metrics, prediction, gold):
    cur_sp_pred = set(map(tuple, prediction))
    gold_sp_pred = set(map(tuple, gold))
    tp, fp, fn = 0, 0, 0
    for e in cur_sp_pred:
        if e in gold_sp_pred:
            tp += 1
        else:
            fp += 1
    for e in gold_sp_pred:
        if e not in cur_sp_pred:
            fn += 1
    prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
    em = 1.0 if fp + fn == 0 else 0.0
    metrics["sp_em"] += em
    metrics["sp_f1"] += f1
    metrics["sp_prec"] += prec
    metrics["sp_recall"] += recall
    return em, prec, recall


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def update_answer(metrics, prediction, gold):
    em = exact_match_score(prediction, gold)
    f1, prec, recall = f1_score(prediction, gold)
    metrics["em"] += float(em)
    metrics["f1"] += f1
    metrics["prec"] += prec
    metrics["recall"] += recall
    return em, prec, recall


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if (
        normalized_prediction in ["yes", "no", "noanswer"]
        and normalized_prediction != normalized_ground_truth
    ):
        return ZERO_METRIC
    if (
        normalized_ground_truth in ["yes", "no", "noanswer"]
        and normalized_prediction != normalized_ground_truth
    ):
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
            if pred.lower() == refer.lower():
                return True
        except Exception:
            pass
        return False

    def score(self, predictions, references):
        """Calculate accuracy."""
        if len(predictions) != len(references):
            return {"error": "preds and refrs have different length"}

        metrics = {
            "em": 0,
            "f1": 0,
            "prec": 0,
            "recall": 0,
            "sp_em": 0,
            "sp_f1": 0,
            "sp_prec": 0,
            "sp_recall": 0,
            "joint_em": 0,
            "joint_f1": 0,
            "joint_prec": 0,
            "joint_recall": 0,
        }
        for dp in references:
            cur_id = dp["_id"]
            can_eval_joint = True
            if cur_id not in predictions["answer"]:
                print("missing answer {}".format(cur_id))
                can_eval_joint = False
            else:
                em, prec, recall = update_answer(
                    metrics, predictions["answer"][cur_id], dp["answer"]
                )
            if cur_id not in predictions["sp"]:
                print("missing sp fact {}".format(cur_id))
                can_eval_joint = False
            else:
                sp_em, sp_prec, sp_recall = update_sp(
                    metrics, predictions["sp"][cur_id], dp["supporting_facts"]
                )

            if can_eval_joint:
                joint_prec = prec * sp_prec
                joint_recall = recall * sp_recall
                if joint_prec + joint_recall > 0:
                    joint_f1 = (
                        2 * joint_prec * joint_recall / (joint_prec + joint_recall)
                    )
                else:
                    joint_f1 = 0.0
                joint_em = em * sp_em

                metrics["joint_em"] += joint_em
                metrics["joint_f1"] += joint_f1
                metrics["joint_prec"] += joint_prec
                metrics["joint_recall"] += joint_recall

        N = len(references)
        for k in metrics.keys():
            metrics[k] /= N

        return metrics


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
