import numpy as np
import utils.var as var
import re
import evaluate
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score
accuracy = evaluate.load("accuracy")

def outer_computer_metrics(args, id2label=None):
    if args.label == 0:
        def compute_metrics_simplelabel(eval_pred, normalize=True, sample_weight=None, weights='quadratic', id=True):
            """
            :param eval_pred:  a tuple of (predictions, labels)
            :param normalize: True
            :param sample_weight: None
            :return:
            """
            predictions, labels = eval_pred
            #Check if predictions is a tuple
            if isinstance(predictions, tuple):
                predictions = predictions[0]
            if len(predictions.shape) > 1:
                predictions = np.argmax(predictions, axis=1)
            accuracy = float(
                    accuracy_score(labels, predictions, normalize=normalize, sample_weight=sample_weight))
            kappa = float(cohen_kappa_score(labels, predictions, sample_weight=sample_weight, weights=weights))
            if kappa < 0:
                kappa =  0.01
            result = {
                "accuracy": accuracy,
                "kappa": kappa
            }
            # result = accuracy.compute(predictions=predictions, references=labels)
            return result
        return compute_metrics_simplelabel
    elif args.label == 1:
        """
        detail label 
        """
        id2simplelabel = {k: int(re.sub(r"\D", "", v)) for k,v in id2label.items()}
        def compute_metrics_detaillabel(eval_pred, normalize=True, sample_weight=None, weights='quadratic', id=True):
            """
            :param eval_pred:  a tuple of (predictions, labels)
            :param normalize: True
            :param sample_weight: None
            :return:
            """
            predictions, labels = eval_pred
            if isinstance(predictions, tuple):
                predictions = predictions[0]
            if len(predictions[0]) > 1:
                predictions = np.argmax(predictions, axis=1)

            """
            Modified the predictions and labels into simpleLabel scale 
            """
            if id:
                predictions = list(map(lambda x: id2simplelabel[x], predictions))
                labels = list(map(lambda x: id2simplelabel[x], labels))


            result = {
                "accuracy": float(
                    accuracy_score(labels, predictions, normalize=normalize, sample_weight=sample_weight)),
                "kappa": float(cohen_kappa_score(labels, predictions, sample_weight=sample_weight, weights=weights))
            }
            return result
        return compute_metrics_detaillabel
    elif args.label == 2:
        id2simplelabel = {k: int(re.sub(r"\D", "", v)) for k, v in id2label.items()}
        def compute_metrics_reducedlabel(eval_pred, normalize=True, sample_weight=None, weights='quadratic', id=True):
            """
            :param eval_pred:  a tuple of (predictions, labels)
            :param normalize: True
            :param sample_weight: None
            :return:
            """
            def transfer(x):
                #todo update for all type 2 current only work for "age"
                p, e = x
                if p == 3 and e == 0:
                    return 2
                return p
            predictions, labels, other_info = eval_pred
            labels = other_info[var.EVAL_LABEL]
            #Check if predictions is a tuple
            if isinstance(predictions, tuple):
                predictions = predictions[0]
            if len(predictions.shape) > 1:
                predictions = np.argmax(predictions, axis=1)
            est_labels = other_info[var.EST_SCORE]

            if id:
                predictions = list(map(lambda x: id2simplelabel[x], predictions))
            predictions = list(map(lambda x: transfer(x), zip(predictions, est_labels)))
            result = {
                "accuracy": float(
                    accuracy_score(labels, predictions, normalize=normalize, sample_weight=sample_weight)),
                "kappa": float(cohen_kappa_score(labels, predictions, sample_weight=sample_weight, weights=weights))
            }
            # result = accuracy.compute(predictions=predictions, references=labels)
            return result
        return compute_metrics_reducedlabel








