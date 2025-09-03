import numpy as np


def metric(y_true, y_pred, amounts, p=0.25, fraud_loss=1.0):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    M = np.asarray(amounts).astype(float)

    tn = (y_true == 0) & (y_pred == 0)
    fp = (y_true == 0) & (y_pred == 1)
    tp = (y_true == 1) & (y_pred == 1)
    fn = (y_true == 1) & (y_pred == 0)

    profit = 0.0
    profit += np.sum(p * M[tn])
    profit += np.sum(-p * M[fp])
    profit += np.sum(fraud_loss * M[tp])
    profit += np.sum(-fraud_loss * M[fn])
    return float(profit)