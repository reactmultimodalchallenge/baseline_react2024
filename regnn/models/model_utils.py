import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np

class ActNorm2D_N(nn.Module):
    # B, N, D -> (:, N, :)
    def  __init__(self, num_channels, eps=1e-5):
        super(ActNorm2D_N, self).__init__()
        self.eps = eps
        self.num_channels = num_channels
        self._log_scale = nn.Parameter(torch.Tensor(num_channels))
        self._shift = nn.Parameter(torch.Tensor(num_channels))
        self._init = False

    def log_scale(self):
        return self._log_scale[None, :, None]

    def shift(self):
        return self._shift[None, :, None]

    def forward(self, x):
        if not self._init:
            with torch.no_grad():
                # initialize params to input stats
                assert self.num_channels == x.size(1)
                mean = torch.transpose(x, 0, 1).contiguous().view(self.num_channels, -1).mean(dim=1)
                zero_mean = x - mean[None, :, None]
                var = torch.transpose(zero_mean ** 2, 0, 1).contiguous().view(self.num_channels, -1).mean(dim=1)
                std = (var + self.eps) ** .5
                log_scale = torch.log(1. / std)
                self._shift.data = - mean * torch.exp(log_scale)
                self._log_scale.data = log_scale
                self._init = True
        log_scale = self.log_scale()
        logdet = log_scale.sum() * x.size(2)
        return x * torch.exp(log_scale) + self.shift(), logdet

    def inverse(self, x):
        return (x - self.shift()) * torch.exp(-self.log_scale())

class ActNorm2D_D(nn.Module):
    # B, N, D -> (:, :, D)
    def  __init__(self, num_channels, eps=1e-5):
        super(ActNorm2D_D, self).__init__()
        self.eps = eps
        self.num_channels = num_channels
        self._log_scale = nn.Parameter(torch.Tensor(num_channels))
        self._shift = nn.Parameter(torch.Tensor(num_channels))
        self._init = False

    def log_scale(self):
        return self._log_scale[None, :, None]

    def shift(self):
        return self._shift[None, :, None]

    def forward(self, x):
        # x: B 25 50
        if not self._init:
            with torch.no_grad():
                # initialize params to input stats
                assert self.num_channels == x.size(1)
                mean = x.view(-1, 50).mean(dim=0)
                # 1 50
                zero_mean = x - mean[None, None, :]
                var = torch.transpose(zero_mean ** 2, 0, 1).contiguous().view(self.num_channels, -1).mean(dim=1)
                std = (var + self.eps) ** .5
                log_scale = torch.log(1. / std)
                self._shift.data = - mean * torch.exp(log_scale)
                self._log_scale.data = log_scale
                self._init = True
        log_scale = self.log_scale()
        logdet = log_scale.sum() * x.size(2)
        return x * torch.exp(log_scale) + self.shift(), logdet

    def inverse(self, x):
        return (x - self.shift()) * torch.exp(-self.log_scale())


class ActNorm2D_ND(nn.Module):
    # B, N, D -> (:, N, :)
    def __init__(self, num_channels, num_dims, eps=1e-5):
        super(ActNorm2D_ND, self).__init__()
        self.eps = eps
        self.num_channels = num_channels
        self.num_dims = num_dims
        self._log_scale = nn.Parameter(torch.Tensor(num_channels, num_dims))
        self._shift = nn.Parameter(torch.Tensor(num_channels, num_dims))
        self._init = False

    def log_scale(self):
        return self._log_scale[None, :, :]

    def shift(self):
        return self._shift[None, :, :]

    def forward(self, x):
        if not self._init:
            with torch.no_grad():
                # initialize params to input stats
                assert self.num_channels == x.size(1)
                assert self.num_dims == x.size(2)
                mean = x.mean(dim=0, keepdim=True)
                zero_mean = x - mean[None, :, :]
                var = torch.mean(zero_mean ** 2, dim=0, keepdim=True)
                std = (var + self.eps) ** .5
                log_scale = torch.log(1. / std)
                self._shift.data = - mean * torch.exp(log_scale)
                self._log_scale.data = log_scale
                self._init = True
        log_scale = self.log_scale()
        logdet = log_scale.sum()
        return x * torch.exp(log_scale) + self.shift(), logdet

    def inverse(self, x):
        return (x - self.shift()) * torch.exp(-self.log_scale())


import math

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    # 也可以判断是否为conv2d，使用相应的初始化方式
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
     # 是否为批归一化层
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class MultiNodeMlp(nn.Module):
    def __init__(self, in_dim, out_dim, n_nodes=25, norm=True):
        super().__init__()
        self.norm = norm
        self.weight1 = nn.Parameter(torch.ones((n_nodes, in_dim, in_dim)))
        self.bias1 = nn.Parameter(torch.zeros((n_nodes, 1, in_dim)))
        self.act_layer = nn.GELU()
        self.weight2 = nn.Parameter(torch.ones((n_nodes, in_dim, out_dim)))
        self.bias2 = nn.Parameter(torch.zeros((n_nodes, 1, out_dim)))
        init.kaiming_uniform_(self.weight1, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight2, a=math.sqrt(5))

    def forward(self, inputs):
        # inputs: B 25 frame_size           weight: 25 frame_size out_dim
        if self.norm:
            self.spectual_norm()
        inputs = inputs.unsqueeze(2)
        outputs = torch.matmul(inputs, self.weight1) + self.bias1
        outputs = outputs.squeeze(2)
        outputs = self.act_layer(outputs).unsqueeze(2)
        outputs = torch.matmul(outputs, self.weight2) + self.bias2

        return outputs.squeeze(2)

    def spectual_norm(self):
        with torch.no_grad():
            norm1 = torch.linalg.norm(self.weight1, dim=(1, 2), ord=2, keepdim=True).to(self.weight1.device)
            norm2 = torch.linalg.norm(self.weight2, dim=(1, 2), ord=2, keepdim=True).to(self.weight2.device)
            # print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
            # print(norm1.shape, norm2.shape)
            # print(self.weight1.shape, self.weight2.shape)
            # print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
            self.weight1.data = self.weight1 / norm1
            self.weight2.data = self.weight2 / norm2


class MLPLayer(nn.Module):
    def __init__(self, in_dim, out_dim, iters=5, n_nodes=25):
        super().__init__()
        self.weight1 = nn.Parameter(torch.ones((n_nodes, in_dim, in_dim)))
        self.bias1 = nn.Parameter(torch.zeros((n_nodes, 1, in_dim)))
        self.act_layer = nn.GELU()
        self.weight2 = nn.Parameter(torch.ones((n_nodes, in_dim, out_dim)))
        self.bias2 = nn.Parameter(torch.zeros((n_nodes, 1, out_dim)))
        self.iters = iters
        init.kaiming_uniform_(self.weight1, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight2, a=math.sqrt(5))

    def forward(self, inputs, norm=True, **kwargs):
        # inputs: B 25 frame_size           weight: 25 frame_size out_dim
        if True:
            self.spectual_norm()

        outputs = self.resforward(inputs)

        return inputs + outputs, 0.

    def resforward(self, inputs, ):
        inputs = inputs.unsqueeze(2)
        outputs = torch.matmul(inputs, self.weight1) + self.bias1
        outputs = outputs.squeeze(2)
        outputs = self.act_layer(outputs).unsqueeze(2)
        outputs = torch.matmul(outputs, self.weight2) + self.bias2

        return outputs.squeeze(2)

    def inverse(self, outputs, **kwargs):
        inputs = outputs
        for i in range(self.iters):
            fx = self.resforward(inputs)
            if (inputs + fx == outputs).all():
                break
            inputs = outputs - fx

        k = torch.abs(inputs - (outputs - self.resforward(inputs)))
        v = (k > 1e-1).sum()

        if v > 0:
            print(torch.linalg.norm(self.weight1, dim=(1, 2), ord=2, keepdim=True).to(self.weight1.device))
            print(torch.linalg.norm(self.weight2, dim=(1, 2), ord=2, keepdim=True).to(self.weight2.device))
            print(v)
            print(k.sum())
            print(k[k>1e-1])
        assert v == 0, 'Invalid Inverse Results'

        return inputs

    def spectual_norm(self):
        with torch.no_grad():
            norm1 = torch.linalg.norm(self.weight1, dim=(1, 2), ord=2, keepdim=True).to(self.weight1.device)
            norm2 = torch.linalg.norm(self.weight2, dim=(1, 2), ord=2, keepdim=True).to(self.weight2.device)
            # print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
            # print(norm1.shape, norm2.shape)
            # print(self.weight1.shape, self.weight2.shape)
            # print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
            self.weight1.data = 0.99 * self.weight1 / norm1
            self.weight2.data = 0.99 * self.weight2 / norm2


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TestMlp(nn.Module):
    def __init__(self, n_layers=6, in_dim=25):
        super().__init__()
        layers = []
        for _ in range(6):
            layer = MultiNodeMlp(in_dim=in_dim, out_dim=in_dim)
            layers.append(layer)

        self.layers = nn.Sequential(*layers)

    def forward(self, inputs):
        outputs = self.layers(inputs)
        return outputs

from .swin_transformer import PatchEmbed
from .cognitive import EdgeLayer

class ModelBefore(nn.Module):
    def __init__(self, img_size=40, in_chans=3, patch_size=8, embed_dim=50,
                 mlp_before=False, n_channels=8, bias=False, neighbors=5):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size=img_size, in_chans=in_chans,
                                      patch_size=patch_size, embed_dim=embed_dim)
        self.edge_layer = EdgeLayer(dim=embed_dim, mlp_before=mlp_before, n_channels=n_channels,
                                    bias=bias, neighbors=neighbors)

    def forward(self, inputs):
        outputs = self.patch_embed(inputs)
        edge = self.edge_layer(outputs)

        return outputs, edge

def log_det_fn(x):
    return torch.logdet(torch.eye(x.size(0)).cuda() + x)


def compute_log_det(inputs, outputs):
    # inputs  B C H W
    # outputs B C H W
    batch_size = outputs.size(0)
    # outVector: C*H*W
    outVector = torch.sum(outputs, 0).view(-1)
    outdim = outVector.size()[0]
    # 雅克比矩阵是 B个 (C*H*W, C*H*W)
    jac = torch.stack([torch.autograd.grad(outVector[i], inputs,
                                     retain_graph=True, create_graph=True)[0].reshape(batch_size, outdim) for i in range(outdim)], dim=1)
    # 最后得到B个logdet(jac)
    log_det = torch.stack([log_det_fn(jac[i,:,:]) for i in range(batch_size)], dim=0)
    return log_det


def power_series_full_jac_exact_trace(Fx, x, k):
    """
    Fast-boi Tr(Ln(d(Fx)/dx)) using power-series approximation with full
    jacobian and exact trace

    :param Fx: output of f(x)
    :param x: input
    :param k: number of power-series terms  to use
    :return: Tr(Ln(I + df/dx))
    """
    _, jac = compute_log_det(x, Fx)
    jacPower = jac
    summand = torch.zeros_like(jacPower)
    for i in range(1, k + 1):
        if i > 1:
            jacPower = torch.matmul(jacPower, jac)
        if (i + 1) % 2 == 0:
            summand += jacPower / (np.float(i))
        else:
            summand -= jacPower / (np.float(i))
    trace = torch.diagonal(summand, dim1=1, dim2=2).sum(1)
    return trace


def power_series_matrix_logarithm_trace(Fx, x, k, n):
    """
    Fast-boi Tr(Ln(d(Fx)/dx)) using power-series approximation
    biased but fast
    :param Fx: output of f(x)
    :param x: input
    :param k: number of power-series terms  to use
    :param n: number of Hitchinson's estimator samples
    :return: Tr(Ln(I + df/dx))
    """
    # trace estimation including power series
    outSum = Fx.sum(dim=0)
    dim = list(outSum.shape)
    dim.insert(0, n)
    dim.insert(0, x.size(0))
    u = torch.randn(dim).to(x.device)
    trLn = 0
    for j in range(1, k + 1):
        if j == 1:
            vectors = u
        # compute vector-jacobian product
        vectors = [torch.autograd.grad(Fx, x, grad_outputs=vectors[:, i],
                                       retain_graph=True, create_graph=True)[0] for i in range(n)]
        # compute summand
        vectors = torch.stack(vectors, dim=1)
        vjp4D = vectors.view(x.size(0), n, 1, -1)
        u4D = u.view(x.size(0), n, -1, 1)
        summand = torch.matmul(vjp4D, u4D)
        # add summand to power series
        if (j + 1) % 2 == 0:
            trLn += summand / np.float(j)
        else:
            trLn -= summand / np.float(j)
    trace = trLn.mean(dim=1).squeeze()
    return trace