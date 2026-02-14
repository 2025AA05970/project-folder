from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)
import numpy as np

def compute_metrics(y_true, y_pred, y_prob):

    y_prob = np.array(y_prob)

    if y_prob.ndim == 1:
        auc = roc_auc_score(y_true, y_prob)

    elif y_prob.shape[1] == 2:
        auc = roc_auc_score(y_true, y_prob[:, 1])

    else:
        auc = roc_auc_score(y_true, y_prob, multi_class="ovr")

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, average="weighted"),
        "Recall": recall_score(y_true, y_pred, average="weighted"),
        "F1": f1_score(y_true, y_pred, average="weighted"),
        "MCC": matthews_corrcoef(y_true, y_pred)
    }
