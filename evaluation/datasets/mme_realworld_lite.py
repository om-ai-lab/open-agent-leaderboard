from evaluation.base import BaseEvaluator
import re


def extract_characters_regex(s, choices=['(A)', '(B)', '(C)', '(D)', '(E)']):
    if type(s) is dict:
        s = ''
    s = s.strip()
    answer_prefixes = [
        'The best answer is',
        'The correct answer is',
        'The answer is',
        'The answer',
        'The best option is'
        'The correct option is',
        'Best answer:'
        'Best option:',
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, '')

    if len(s.split()) > 10 and not re.search('[ABCDE]', s):
        return ''
    matches = re.search(r'[ABCDE]', s)
    if matches is None:
        for choice in choices:
            if s.lower() in choice.lower():
                return choice[1]
        return ''
    return matches[0]

class MMERealWorldLiteEvaluator(BaseEvaluator):
        
    def is_equal(self, pred, refer):
        try:
            pred_postprocess = extract_characters_regex(pred)
            if pred_postprocess.lower() == refer.lower():
                return True
        except Exception:
            pass
        return False
    
    def score(self, predictions, references):
        """Calculate accuracy."""
        if len(predictions) != len(references):
            return {'error': 'preds and refrs have different length'}

        correct = 0
        count = 0
        details = []
        for i, j in zip(predictions, references):
            detail = {'pred': i, 'answer': j, 'correct': False}
            count += 1
            if self.is_equal(i, j):
                correct += 1
                detail['correct'] = True
            details.append(detail)
        result = {'accuracy': 100 * correct / count, 'details': details}
        return result
    