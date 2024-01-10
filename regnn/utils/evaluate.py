import torch
import numpy as np
from numpy import array, zeros, full, argmin, inf, ndim
from scipy.spatial.distance import cdist
from torch.nn import functional as F
from torchmetrics.functional import concordance_corrcoef

def CCC(ground_truthes, predictions):
    length = predictions.shape[1]
    ccces = 0.0
    count = 0
    for i in range(length):
        ground_truth = ground_truthes[:, i].reshape(-1, 1)
        prediction = predictions[:, i].reshape(-1, 1)
        mean_gt = torch.mean(ground_truth)
        mean_pred = torch.mean(prediction)
        var_gt = torch.var(ground_truth)
        var_pred = torch.var(prediction)
        v_pred = prediction - mean_pred
        v_gt = ground_truth - mean_gt
        cor = torch.sum(v_pred * v_gt) / (torch.sqrt(torch.sum(v_pred ** 2)) * torch.sqrt(torch.sum(v_gt ** 2)) + 1e-8)
        sd_gt = torch.std(ground_truth)
        sd_pred = torch.std(prediction)
        numerator = 2 * cor * sd_gt * sd_pred
        denominator = var_gt + var_pred + (mean_gt - mean_pred) ** 2
        ccc = numerator / denominator
        ccces += torch.abs(ccc)
        if torch.abs(ccc) > 0.:
            count += 1

    return ccces / count

def ccc(target, pred):
    # for i in range(25):
    #     c = torch.abs(concordance_corrcoef(target[:, i], pred[:, i]))
    #     print(c)
    #     print(target[:, i], pred[:, i])
    #     print('='*10)
    return torch.nanmean(torch.abs(concordance_corrcoef(target, pred)))

def person_r(labels, preds):
    length = labels.shape[1]
    assert length == 25, "Length != 25"
    pcc = 0.
    for i in range(length):
        label, pred = labels[:, i], preds[:, i]
        mean_label = torch.mean(label, dim=-1, keepdims=True)
        mean_pred = torch.mean(pred, dim=-1, keepdims=True)
        new_label, new_pred = label - mean_label, pred - mean_pred
        # label_norm = F.normalize(new_label, p=2, dim=1)
        # pred_norm = F.normalize(new_pred, p=2, dim=1)

        value = F.cosine_similarity(new_label, new_pred, dim=-1)
        pcc += value

    return pcc.mean()

def pcc(x, y):
    assert x.shape == y.shape
    x = x.transpose(1, 0)
    y = y.transpose(1, 0)
    centered_x = x - x.mean(dim=-1, keepdim=True)
    centered_y = y - y.mean(dim=-1, keepdim=True)

    covariance = (centered_x * centered_y).sum(dim=-1, keepdim=True)

    bessel_corrected_covariance = covariance / (x.shape[-1] - 1)

    x_std = x.std(dim=-1, keepdim=True)
    y_std = y.std(dim=-1, keepdim=True)

    corr = bessel_corrected_covariance / ((x_std * y_std) + 1e-8)
    return torch.abs(corr).sum() / torch.nonzero(corr).shape[0]

def accelerated_dtw(x, y, dist, warp=1):
    assert len(x)
    assert len(y)
    if np.ndim(x) == 1:
        x = x.reshape(-1, 1)
    if np.ndim(y) == 1:
        y = y.reshape(-1, 1)
    r, c = len(x), len(y)
    D0 = np.zeros((r + 1, c + 1))
    D0[0, 1:] = inf
    D0[1:, 0] = inf
    D1 = D0[1:, 1:]
    D0[1:, 1:] = cdist(x, y, dist)
    C = D1.copy()
    for i in range(r):
        for j in range(c):
            min_list = [D0[i, j]]
            for k in range(1, warp + 1):
                min_list += [D0[min(i + k, r), j],
                             D0[i, min(j + k, c)]]
            D1[i, j] += min(min_list)

    return D1[-1, -1]

def tlcc(x, y, lag):
    """
    计算时间序列x和y之间的Time Lagged Cross-Correlation（TLCC）。

    Args：
    x: torch.Tensor，形状为(n,)的时间序列数据。
    y: torch.Tensor，形状为(n,)的时间序列数据。
    lag: int，时间滞后量。滞后量可以是正数（y滞后于x）或负数（y超前于x）。

    Returns：
    TLCC值。
    """
    # 计算x和y的平均值
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)

    # 计算x和y的标准差
    std_x = torch.std(x)
    std_y = torch.std(y)

    # 计算x和y的标准化时间序列
    x_norm = (x - mean_x) / std_x
    y_norm = (y - mean_y) / std_y

    # 将y_norm向右平移lag个单位
    if lag > 0:
        y_norm = torch.cat([torch.zeros(lag), y_norm[:-lag]])
    elif lag < 0:
        y_norm = torch.cat([y_norm[-lag:], torch.zeros(-lag)])

    # 计算x_norm和y_norm的TLCC
    tlcc_value = torch.corrcoef(x_norm, y_norm)[0, 1]
    return tlcc_value


def s_mse(preds):
    # preds: (10, 750, 25)
    preds_ = preds.reshape(preds.shape[0], -1)
    dist = torch.pow(torch.cdist(preds_, preds_), 2)
    dist = torch.sum(dist) / (preds.shape[0] * (preds.shape[0] - 1))
    return dist / preds_.shape[1]


def FRVar(preds):
    if len(preds.shape) == 3:
        # preds: (10, 750, 25)
        var = torch.var(preds, dim=1)
        return torch.mean(var)
    elif len(preds.shape) == 4:
        # preds: (N, 10, 750, 25)
        var = torch.var(preds, dim=2)
        return torch.mean(var)


def FRDvs(preds):
    # preds: (N, 10, 750, 25)
    preds_ = preds.reshape(preds.shape[0], preds.shape[1], -1)
    preds_ = preds_.transpose(0, 1)
    # preds_: (10, N, 750*25)
    dist = torch.pow(torch.cdist(preds_, preds_), 2)
    # dist: (10, N, N)
    dist = torch.sum(dist) / (preds.shape[0] * (preds.shape[0] - 1) * preds.shape[1])
    return dist / preds_.shape[-1]


def crosscorr(datax, datay, lag=0, dim=25):
    pcc_list = []
    for i in range(dim):
        # print(datax.shape, datay.shape)
        cn_1, cn_2 = shift(datax[:, i], datay[:, i], lag)
        pcc_i = torch.corrcoef(torch.stack([cn_1, cn_2], dim=0))[0, 1]
        pcc_list.append(pcc_i.item())
    return torch.mean(torch.Tensor(pcc_list))


def calculate_tlcc(pred, sp, seconds=2, fps=25):
    rs = [crosscorr(pred, sp, lag, sp.shape[-1]) for lag in range(-int(seconds * fps - 1), int(seconds * fps))]
    peak = max(rs)
    center = rs[len(rs) // 2]
    offset = len(rs) // 2 - torch.argmax(torch.Tensor(rs))
    return peak, center, offset

def TLCC(pred, speaker):
    # pred: N 10 750 25
    # speaker: N 750 25
    offset_list = []
    for k in range(speaker.shape[0]):
        pred_item = pred[k]
        sp_item = speaker[k]
        for i in range(pred_item.shape[0]):
            peak, center, offset = calculate_tlcc(pred_item[i].float(), sp_item.float())
            offset_list.append(torch.abs(offset).item())
    return torch.mean(torch.Tensor(offset_list)).item()


def SingleTLCC(pred, speaker):
    # pred: 10 750 25
    # speaker: 750 25
    offset_list = []
    for i in range(pred.shape[0]):
        peak, center, offset = calculate_tlcc(pred[i].float(), speaker.float())
        offset_list.append(torch.abs(offset).item())
    return torch.mean(torch.Tensor(offset_list)).item()


def shift(x, y, lag):
    if lag > 0:
        return x[lag:], y[:-lag]
    elif lag < 0:
        return x[:lag], y[-lag:]
    else:
        return x, y


import pandas as pd


def shift_elements(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result


def crosscorr_(datax, datay, lag=0, dim=25, wrap=False):
    pcc_list = []
    for i in range(25):
        cn_1 = pd.Series(datax[:, i].reshape(-1))
        cn_2 = pd.Series(datay[:, i].reshape(-1))
        pcc_i = cn_1.corr(cn_2.shift(lag))
        pcc_list.append(pcc_i)
    return np.mean(pcc_list)


def calculate_tlcc_(pred, sp, seconds=2, fps=25):
    rs = [crosscorr_(pred, sp, lag, sp.shape[-1]) for lag in range(-int(seconds * fps - 1), int(seconds * fps))]
    peak = max(rs)
    center = rs[len(rs) // 2]
    offset = len(rs) // 2 - np.argmax(rs)
    return peak, center, offset


def compute_TLCC(pred, speaker):
    offset_list = []
    for k in range(speaker.shape[0]):
        pred_item = pred[k]
        sp_item = speaker[k]
        for i in range(pred_item.shape[0]):
            peak, center, offset = calculate_tlcc_(pred_item[i].numpy().astype(np.float32),
                                                   sp_item.numpy().astype(np.float32))
            offset_list.append(np.abs(offset))
    return np.mean(offset_list)


def compute_singel_tlcc(pred, speaker):
    pred = pred.detach().cpu().numpy()
    speaker = speaker.detach().cpu().numpy()
    peak, center, offset = calculate_tlcc_(pred.astype(np.float32),
                                          speaker.astype(np.float32))

    return np.abs(offset)




