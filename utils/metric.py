import numpy as np
import re
import evaluate
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score
accuracy = evaluate.load("accuracy")

def outer_computer_metrics(args, id2label=None):
    if args.label == 0:
        def compute_metrics_simplelabel(eval_pred, normalize=True, sample_weight=None):
            """
            :param eval_pred:  a tuple of (predictions, labels)
            :param normalize: True
            :param sample_weight: None
            :return:
            """
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            result = {
                "accuracy": float(
                    accuracy_score(labels, predictions, normalize=normalize, sample_weight=sample_weight)),
                "kappa": float(cohen_kappa_score(labels, predictions, sample_weight=sample_weight))
            }
            # result = accuracy.compute(predictions=predictions, references=labels)
            return result
        return compute_metrics_simplelabel
    elif args.label == 1:
        """
        detail label 
        """
        id2simplelabel = {k: int(re.sub(r"\D", "", v)) for k,v in id2label.items()}
        def compute_metrics_detaillabel(eval_pred, normalize=True, sample_weight=None):
            """
            :param eval_pred:  a tuple of (predictions, labels)
            :param normalize: True
            :param sample_weight: None
            :return:
            """
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)

            """
            Modfied the predictions and labels into simpleLabel scale 
            """
            predictions = list(map(lambda x: id2simplelabel[x], predictions))
            labels = list(map(lambda x: id2simplelabel[x], labels))


            result = {
                "accuracy": float(
                    accuracy_score(labels, predictions, normalize=normalize, sample_weight=sample_weight)),
                "kappa": float(cohen_kappa_score(labels, predictions, sample_weight=sample_weight))
            }
            return result
        return compute_metrics_detaillabel







