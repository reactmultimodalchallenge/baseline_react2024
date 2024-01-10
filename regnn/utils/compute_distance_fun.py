from tslearn.metrics import dtw
import numpy as np
import torch

def compute_distance(a, b):
    if not type(a) == np.ndarray:
        a = a.clone().detach().cpu().numpy()
        b = b.clone().detach().cpu().numpy()
    a = a.reshape(-1, 25)
    b = b.reshape(-1, 25)
    res = 0
    for st, ed, weight in [(0, 15, 1 / 15), (15, 17, 1), (17, 25, 1 / 8)]:
        res += weight * dtw(a[ : , st : ed], b[ : , st : ed])
    return res


def compute_total_distance(preds, targets):
    # preds: 10 750 25
    # targets: N 750 25
    if not type(preds) == np.ndarray:
        preds = preds.clone().detach().cpu().numpy()
    if not type(targets) == np.ndarray:
        targets = targets.clone().detach().cpu().numpy()

    dtw_lists = []
    for i in range(preds.shape[0]):
        for j in range(targets.shape[0]):
            pred = preds[i]
            target = targets[j]
            res = 0.
            for st, ed, weight in [(0, 15, 1 / 15), (15, 17, 1), (17, 25, 1 / 8)]:
                res += weight * dtw(pred[ : , st : ed], target[ : , st : ed])
            dtw_lists.append(res)

    return min(dtw_lists)
