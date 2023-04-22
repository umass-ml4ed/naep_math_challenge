import numpy as np
import evaluate
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score
accuracy = evaluate.load("accuracy")
def compute_metrics(eval_pred, normalize=True, sample_weight=None):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    result = {
        "accuracy": float( accuracy_score(labels, predictions, normalize=normalize, sample_weight=sample_weight)),
        "kappa": float(cohen_kappa_score(labels, predictions, sample_weight=sample_weight))
    }
    #result = accuracy.compute(predictions=predictions, references=labels)
    return result