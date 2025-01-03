from evaluation.base import BaseEvaluator

class AQuAEvaluator(BaseEvaluator):
        
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
    
if __name__ == '__main__':
    import os
    import json

    evaluator = AQuAEvaluator()
    # Example usage
    
    outputs_path = "/path/to/your/local"
    with open(os.path.join(outputs_path, 'aqua.json'), 'r', encoding='utf-8') as file:
        data = json.load(file)['model_result']
        
        predictions = [item['last_output'].split("The answer is")[-1].strip() for item in data]
        references = [item['ground_truth'] for item in data]
        
    result = evaluator.score(predictions, references)
    print(result)