import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score


def metrics(y_test, y_pred):
    score = float(sum(y_pred == y_test)) / float(len(y_test))
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, average='weighted', labels=np.unique(y_pred))
    recall = recall_score(
        y_test, y_pred, average='weighted', labels=np.unique(y_pred))

    metrics = [accuracy, precision, recall]

    return metrics


def compare(our_metrics, sklearn_metrics):
    res = pd.DataFrame([[our_metrics[0], sklearn_metrics[0]],
                        [our_metrics[1], sklearn_metrics[1]],
                        [our_metrics[2], sklearn_metrics[2]]],
                       ['Accuracy', 'Precision', 'Recall'],
                       ['Our Implementation', 'Sklearn\'s Implementation'])
    return res
