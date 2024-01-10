import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.distributions.multivariate_normal import MultivariateNormal
# from sdtw.distance import SquaredEuclidean
# from .softdtw import SoftDTW
from utils.evaluate import person_r
from torchmetrics.functional import pearson_corrcoef

class PCC(nn.Module):
    def __init__(self, ):
        super(PCC, self).__init__()

    def forward(self, labels, preds):
        batch_size = labels.shape[0]
        pcc = 0
        for i in range(batch_size):
            pcc -= person_r(labels[i], preds[i])

        return pcc / batch_size

class PearsonCC(nn.Module):
    def __init__(self):
        super(PearsonCC, self).__init__()

    def forward(self, preds, targets):
        batch_size = preds.shape[0]
        pcces = 0.
        for i in range(batch_size):
            pcces += (1 - torch.nanmean(torch.abs(pearson_corrcoef(preds[i], targets[i]))))

        return pcces / batch_size

class DistributionLoss(nn.Module):
    def __init__(self, dis_type='Gaussian'):
        super().__init__()
        self.dis_type = dis_type

    def cal_MultiGaussian(self, feature, means, var):
        # var = var / var.sum()
        k = means.shape[0]
        # V = torch.eye(var.shape[0], device='cuda:0') + var @ var.transpose(1, 0)
        V = torch.eye(var.shape[0], device='cuda:0')
        log_probs = 0.
        for i in range(k):
            distribution = MultivariateNormal(loc=means[i], covariance_matrix=V)
            log_prob = distribution.log_prob(feature[i] if len(feature.shape) == 2 else feature[:, i])
            log_probs -= log_prob

        return log_probs

    def forward(self, latent_feature, params):
        if self.dis_type == 'MultiGaussian':
            means, vars = params
            vars = torch.diag_embed(vars)
            dist = MultivariateNormal(means, vars)
            log_probs = dist.log_prob(latent_feature)
            return -torch.sum(log_probs)

        elif self.dis_type == 'Gaussian':
            mean, var = params
            n_elements, frame_size = latent_feature.shape[1], latent_feature.shape[2]
            distribution = MultivariateNormal(loc=mean, covariance_matrix=var)
            # loss = -distribution.log_prob(latent_feature)
            latent_feature = latent_feature.reshape(-1, n_elements * frame_size)
            loss = torch.mean((latent_feature - mean)**2 )\
                   # / (var**2 + 1e-6)
                   #            )
            loss_2 = torch.exp(distribution.log_prob(latent_feature))
            # loss = torch.exp(loss)
            return loss, loss_2

        elif self.dis_type == 'Gmm':
            mean, var, muk = params
            log_probs = 0.0
        return log_probs


class AllMseLoss(nn.Module):
    def __init__(self, ):
        super(AllMseLoss, self).__init__()

    def forward(self, preds, targets, lengthes):
        preds = torch.repeat_interleave(preds, dim=0, repeats=torch.tensor(lengthes, device=preds.device))
        assert preds.shape == targets.shape, "Not equal shape"
        return F.mse_loss(preds, targets)


class ThreMseLoss(nn.Module):
    def __init__(self, ):
        super(ThreMseLoss, self).__init__()

    def forward(self, preds, targets, lengthes, threshold):
        batch = preds.shape[0]
        if not lengthes:
            all_mse = F.mse_loss(preds, targets, reduction='none').mean(dim=(1, 2))
            if threshold is not None:
                mask = torch.where(torch.abs(all_mse) >= threshold,
                                   torch.tensor(1, device=preds.device), torch.tensor(0, device=preds.device))
                all_mse = all_mse * mask.float()

            return all_mse.mean()

        preds = torch.repeat_interleave(preds, dim=0, repeats=torch.tensor(lengthes, device=preds.device))
        all_mse = F.mse_loss(preds, targets, reduction='none').mean(dim=(1, 2))
        results = 0.
        start = 0
        for length in lengthes:
            cur_mse = torch.min(all_mse[start:start+length])
            if threshold is None or cur_mse > threshold:
                results += cur_mse

            start += length
        return results / batch

# THis loss is for training the cognitive processor
class AllThreMseLoss(nn.Module):
    def __init__(self, cal_type='min'):
        super(AllThreMseLoss, self).__init__()
        assert cal_type in {'min', 'all'}
        self.cal_type = cal_type

    def forward(self, preds, targets, lengths, threshold):
        # 将预测结果张量重复多次
        preds = torch.repeat_interleave(preds, dim=0, repeats=torch.tensor(lengths, device=preds.device))

        # 计算每个样本与目标结果之间的均方误差
        # print(preds)
        # print(targets)
        all_mse = F.mse_loss(preds, targets, reduction='none').mean(dim=(1, 2))

        # 将所有样本的均方误差按照长度拆分成多个小张量，并计算每个小张量中的最小值
        if self.cal_type == 'min':
            seqs = torch.split(all_mse, lengths)
            mse = torch.stack([torch.min(seq) for seq in seqs])
        else:
            mse = all_mse

        # 根据阈值筛选出符合条件的样本，并计算所有符合条件的样本的均方误差的平均值
        if threshold is not None:
            mse = torch.where(mse > threshold, mse,
                              torch.tensor([0.0], dtype=torch.float, device=mse.device)
                              )

        if self.cal_type == 'min':
            return torch.mean(mse)
        else:
            seqs = torch.split(mse, lengths)
            mse = torch.stack([torch.mean(seq) for seq in seqs])
            return torch.mean(mse)


    # def forward(self, preds, targets, lengthes, threshold):
    #     batch = preds.shape[0]
    #     preds = torch.repeat_interleave(preds, dim=0, repeats=torch.tensor(lengthes, device=preds.device))
    #     assert preds.shape == targets.shape, "Not equal shape"
    #
    #     all_mse = F.mse_loss(preds, targets, reduction='none').mean(dim=(1, 2))
    #     seqs = torch.split(all_mse, lengthes)
    #     min_mse = torch.stack([torch.min(seq, dim=0) for seq in seqs])
        # results = 0.
        # start = 0
        # for length in lengthes:
        #     cur_mse = torch.min(all_mse[start:start+length])
        #     if threshold is None or cur_mse > threshold:
        #         results += cur_mse
        #
        #     start += length
        # return results / batch

# This loss is for training the REGNN. It is self-supervised.
class MidLoss(nn.Module):
    def __init__(self, loss_type='L2'):
        super(MidLoss, self).__init__()
        if loss_type == 'L2':
            self.loss = nn.MSELoss()
        elif loss_type == 'L1':
            self.loss = nn.L1Loss()
        else:
            raise AttributeError

    def forward(self, inputs, lengths):
        # 将输入张量按照长度拆分成多个小张量
        seqs = torch.split(inputs, lengths)

        # 计算每个小张量的平均值，并将结果沿着第1个维度堆叠起来
        means = torch.stack([torch.mean(seq, dim=0) for seq in seqs])

        means = torch.repeat_interleave(means, dim=0, repeats=torch.tensor(lengths, device=means.device))
        # 将平均值张量扩展成与输入张量相同的形状，并计算损失函数值
        loss = self.loss(means, inputs)

        # 计算平均损失函数值
        return loss.mean()

        # start = 0
        # results = 0.
        # batch_size = len(lengthes)
        # for length in lengthes:
        #     sub_inputs = inputs[start:start+length]
        #     mean_val = torch.mean(sub_inputs, dim=1, keepdim=True)
        #     start += length
        #
        #     results += self.loss(mean_val.expand_as(sub_inputs), sub_inputs)
        #
        # return results / batch_size

class S_MSE(nn.Module):
    def __init__(self, loss_type='L2'):
        super(S_MSE, self).__init__()
        self.loss_type = loss_type

    def forward(self, inputs, lengths):
        # 将输入张量按照长度拆分成多个小张量
        seqs = torch.split(inputs, lengths)
        results = []
        for seq in seqs:
            results.append(self.cal(seq))

        results = torch.stack(results, 0)

        return torch.nanmean(results)

    def cal(self, data):
        data_ = data.reshape(data.shape[0], -1)
        if self.loss_type == 'L1':
            dist = torch.pow(torch.cdist(data_, data_), 1)
        else:
            dist = torch.pow(torch.cdist(data_, data_), 2)
        dist = torch.sum(dist) / (data.shape[0] * (data.shape[0] - 1))
        return dist / data_.shape[1]

