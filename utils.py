from scipy.optimize import linear_sum_assignment
import numpy as np
import torch
from tqdm import tqdm

def cluster_acc(y_true, y_pred, reassign:bool = False):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size

    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    cluster_accuracy = w[row_ind, col_ind].sum() / y_pred.size

    if reassign:
        reassignment = dict(zip(row_ind, col_ind))
        return (
            reassignment,
            cluster_accuracy
        )
    else:
        return cluster_accuracy

def predict_cluster_accuracy(model, loader, device):
    model.to(device)
    targets, predicted = [], []
    for x, y in tqdm(loader, desc='Evaluate cluster accuracy'):
        x, y = x.to(device), y.to(device)
        output = model(x)#.reshape(-1, 28*28))

        y_pred = output.argmax(1)

        targets.append(y)
        predicted.append(y_pred)

    targets = torch.cat(targets).cpu().numpy()
    predicted = torch.cat(predicted).cpu().numpy()

    reassignment, accuracy = cluster_acc(targets, predicted, reassign=True)
    return (reassignment, accuracy, predicted)

