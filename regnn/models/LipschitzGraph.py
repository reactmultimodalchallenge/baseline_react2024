import torch
import torchvision
import torch.nn as nn
import numpy as np
import scipy
from .model_utils import MultiNodeMlp, ActNorm2D_N, compute_log_det, MLPLayer, \
    power_series_full_jac_exact_trace, power_series_matrix_logarithm_trace

MULTI_MLP = False
class GraphAttention(nn.Module):
    def __init__(self, num_features, edge_channel, act_type):
        super(GraphAttention, self).__init__()
        self.num_features = num_features
        self.edge_channel = edge_channel
        w_value = torch.ones(1, edge_channel)
        self.w = nn.Parameter(w_value)
        self.qk = nn.Linear(num_features, num_features * 2, bias=True)
        self.scale = num_features ** -0.5
        if MULTI_MLP:
            self.multi_mlp = MultiNodeMlp(in_dim=num_features, out_dim=num_features)
        act_layer = {
            "ReLU": nn.ReLU,
            "ELU": nn.ELU,
            "GeLU": nn.GELU,
            "None": nn.Identity,
        }[act_type]
        self.act_layer = act_layer()

    def get_norm(self, weight):
        with torch.no_grad():
            weight_q, weight_k = weight[:self.num_features, :], weight[self.num_features:, :]
            dot_matrix = weight_q @ weight_k.T * self.scale
            self.lipnorm = torch.linalg.norm(torch.eye(self.num_features,
                                                       device=weight.device) + 2 * dot_matrix,
                                             ord=2) + 5

    def forward(self, x, edge, cal_norm=True):

        B, N, D = x.shape
        x = self.act_layer(x)
        # print('act:', x)
        x = torch.sigmoid(x)
        # print('sigmoid:', x)
        if MULTI_MLP:
            self.multi_mlp.spectual_norm()
            x = self.multi_mlp(x)

        if cal_norm:
            self.get_norm(self.qk.weight.data)

        qk = self.qk(x).reshape(B, N, 2, D).permute(2, 0, 1, 3)
        q, k = qk[0], qk[1]

        # attn B N N
        attn = q @ k.transpose(-2, -1) * self.scale
        attn = attn.softmax(dim=-1)

        # edge P N N -> new_edge B P N N
        # print('attn:', attn[0])
        # print('edge:', edge[0])
        new_edge = attn.unsqueeze(1) * edge
        # print('new_edge1:', new_edge[0])
        new_edge = self.norm_edge(new_edge)
        # print('new_edge2:', new_edge[0])
        new_edge = torch.einsum('bpnd,ip -> bind', new_edge, self.w / torch.sum(self.w)).squeeze(1)

        # message passing x: B N D --> B P N D
        x = torch.matmul(new_edge, x) / self.lipnorm
        # print('final:', x)
        # x: B P N D -> B N D

        return x

    def norm_edge(self, edge):
        # edge B P N N
        norm_row_edge = edge / (torch.sum(edge, dim=-1, keepdims=True))

        return norm_row_edge

import time
class GraphLayer(nn.Module):
    def __init__(self, num_features, num_channels, edge_channel, get_logdets=True,
                 act_type='ELU', delta=0.1, iters=5, norm=False):
        super(GraphLayer, self).__init__()
        self.num_features = num_features
        self.get_logdets = get_logdets
        self.delta = delta
        self.iters = iters
        self.edge_channel = edge_channel
        self.norm = norm
        if norm:
            self.norm_layer = ActNorm2D_N(num_channels=num_channels)
        self.attn_1 = GraphAttention(num_features, edge_channel, act_type=act_type)
        # self.attn_2 = GraphAttention(num_features, edge_channel, act_type=act_type)

    def forward(self, x, edge):
        logdets = 0.
        # norm
        if self.norm:
            x, logdet = self.norm_layer(x)
            # print('norm:', x)
            logdets += logdet if self.get_logdets else 0

        fx = self.attn_1(x, edge)
        # print('fx:', fx)
        # fx = self.attn_2(fx, edge)
        y = x + fx

        if self.get_logdets:
            logdet = power_series_matrix_logarithm_trace(fx, x, 5, 1)
            logdets += logdet.sum()

        return y, logdets

    def inverse(self, y, edge, cal_norm):
        x = y
        for i in range(self.iters):
            cal_norm = (i == 0 and cal_norm)
            fx = self.attn_1(x, edge, cal_norm)
            # fx = self.attn_2(fx, edge, cal_norm)
            sub = (torch.abs(x - (y - fx)) > 1e-3).sum()
            x = y - fx
            if sub == 0:
                # print('Notice:', i)
                break
        # print((torch.abs(y - (x +self.attn_1(x, edge))) > 1e-3).sum())
        assert (torch.abs(y - (x +self.attn_1(x, edge))) > 1e-3).sum() == 0, 'Invalid Inverse Results'
        if self.norm:
            x = self.norm_layer.inverse(x)
        return x



# 输入与输出 shape: B 25 50
class LipschitzGraph(nn.Module):
    def __init__(self, edge_channel=8, num_channels=25, num_features=50, n_layers=2,
                 act_type='ReLU', LU=True, norm=False, get_logdets=True):
        super(LipschitzGraph, self).__init__()
        self.n_layers = n_layers
        self.act = act_type
        self.num_features = num_features
        self.get_logdets = get_logdets

        # Layers = [MLPLayer(in_dim=num_features, out_dim=num_features)]
        Layers = []
        for _ in range(n_layers):
            Layers.append(GraphLayer(num_channels=num_channels,
                                                 num_features=num_features,
                                                 edge_channel=edge_channel,
                                                 get_logdets=get_logdets,
                                                 act_type=act_type,
                                                 norm=norm,
                                                 ))

        # Layers.append(MLPLayer(in_dim=num_features, out_dim=num_features))
        self.Layers = nn.Sequential(*Layers)
        w_shape = [num_features, num_features]
        w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype(np.float32)
        self.LU = LU
        if not self.LU:
            # Sample a random orthogonal matrix:
            self.register_parameter("weight", nn.Parameter(torch.Tensor(w_init)))
        else:
            np_p, np_l, np_u = scipy.linalg.lu(w_init)
            np_s = np.diag(np_u)
            np_sign_s = np.sign(np_s)
            np_log_s = np.log(np.abs(np_s))
            np_u = np.triu(np_u, k=1)
            l_mask = np.tril(np.ones(w_shape, dtype=np.float32), -1)
            eye = np.eye(*w_shape, dtype=np.float32)

            self.register_buffer('p', torch.Tensor(np_p.astype(np.float32)))
            self.register_buffer('sign_s', torch.Tensor(np_sign_s.astype(np.float32)))
            self.l = nn.Parameter(torch.Tensor(np_l.astype(np.float32)))
            self.log_s = nn.Parameter(torch.Tensor(np_log_s.astype(np.float32)))
            self.u = nn.Parameter(torch.Tensor(np_u.astype(np.float32)))
            self.l_mask = torch.Tensor(l_mask)
            self.eye = torch.Tensor(eye)
        self.w_shape = w_shape

    def get_weight(self, input, reverse):
        if not self.LU:
            # h*w
            pixels = input.shape[1]
            dlogdet = torch.slogdet(self.weight)[1] * pixels
            if not reverse:
                weight = self.weight
            else:
                weight = torch.inverse(self.weight.double()).float()
            # print(weight.shape)
            return weight, dlogdet
        else:
            self.p = self.p.to(input.device)
            self.sign_s = self.sign_s.to(input.device)
            self.l_mask = self.l_mask.to(input.device)
            self.eye = self.eye.to(input.device)
            l = self.l * self.l_mask + self.eye
            u = self.u * self.l_mask.transpose(0, 1).contiguous() + torch.diag(self.sign_s * torch.exp(self.log_s))
            dlogdet = torch.abs(torch.sum(self.log_s) * input.shape[1])
            if not reverse:
                w = torch.matmul(self.p, torch.matmul(l, u))
            else:
                l = torch.inverse(l.double()).float()
                u = torch.inverse(u.double()).float()
                w = torch.matmul(u, torch.matmul(l, self.p.double().inverse().float()))
            return w, dlogdet

    def forward(self, inputs, edge):
        # inputs B, N, D
        B, N, D = inputs.shape
        logdets = 0.
        x = inputs
        for i, layer in enumerate(self.Layers):
            x, logdet = layer(x, edge=edge)
            logdets = logdets + logdet

        # weight, dlogdet = self.get_weight(x, reverse=False)
        # x = x @ weight
        # logdets += dlogdet

        if self.get_logdets:
            logdets = logdets / self.n_layers

        return x, logdets

    def inverse(self, y, edge, cal_norm=True):
        B, N, D = y.shape
        # weight, _ = self.get_weight(y, reverse=True)
        # y = y @ weight

        for layer in self.Layers[::-1]:
            y = layer.inverse(y, edge=edge, cal_norm=cal_norm)

        return y