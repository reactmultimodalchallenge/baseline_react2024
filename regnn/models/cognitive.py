import torch
import torch.nn as nn
import torchvision
from .distribution import GmmDistribution, MultiGaussian, Gaussian
from .model_utils import Mlp, MultiNodeMlp


class EdgeLayer(nn.Module):
    def __init__(self, dim, mlp_before=False, n_channels=8, bias=False, neighbors=5):
        super().__init__()
        self.n_channels = n_channels
        self.neighbors = neighbors
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = dim ** -0.5
        if not mlp_before:
            self.qk = nn.Linear(dim, dim * 2 * self.n_channels, bias=bias)
        else:
            self.qk = nn.Sequential(Mlp(dim, dim),
                                    nn.Linear(dim, dim * 2 * self.n_channels, bias=bias)
                                    )

    def forward(self, x):
        B, N, C = x.shape
        # B, N, 2, P, C
        #       |
        # 2, B, P, N, C
        qk = self.qk(x).reshape(B, N, 2, self.n_channels, C).permute(2, 0, 3, 1, 4)
        q, k = qk[0], qk[1]
        # B, P, N, N
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        edge = self.clip(attn)

        return edge

    def clip(self, edge, norm=True):
        _edge = edge.detach().clone()
        sum_edge = torch.sum(_edge, dim=1)
        value, index = torch.topk(sum_edge, self.neighbors, dim=-1)
        masks = []
        for i in range(index.shape[0]):
            inds = index[i]
            mask = torch.eye(sum_edge.shape[-1])
            for j, ind in enumerate(inds):
                for ind_ in inds:
                    mask[j][ind_] = 1.

            masks.append(mask)

        masks = torch.stack(masks, dim=0).to(edge.device)

        new_edge = masks.unsqueeze(1) * edge
        if norm:
            new_edge = self.norm_edge(new_edge)
        return new_edge

    def norm_edge(self, edge):
        # edge B P N N
        norm_row_edge = edge / (torch.sum(edge, dim=-1, keepdims=True) + 1e-16)
        norm_col_edge = norm_row_edge / (torch.sum(norm_row_edge, dim=-2, keepdims=True) + 1e-16)
        normed_edge = torch.matmul(norm_row_edge, norm_col_edge.transpose(-1, -2))
        return normed_edge


class CognitiveProcessor(nn.Module):
    def __init__(self, n_elements=25, input_dim=64, convert_type='indirect', mlp_in_edge=True, n_channels=8,
                 num_features=50, k=6, multi_var=False, differ_muk=False, muk_func=False, dis_type='Gmm'):
        super().__init__()
        if convert_type == 'indirect':
            self.convert_layer = MultiNodeMlp(in_dim=input_dim, out_dim=n_elements, n_nodes=50)
            # for i in range(n_elements):
            #     FC_layers = nn.Sequential(nn.Linear(input_dim, input_dim // 2),
            #                               nn.Linear(input_dim // 2, 1)
            #                               )
            #     self.__setattr__(name='fc_' + str(i), value=FC_layers)
        elif convert_type == 'direct':
            self.convert_layer = Mlp(in_features=input_dim, hidden_features=input_dim // 2, out_features=n_elements)
        elif convert_type == 'none':
            self.convert_layer = nn.Identity()
        else:
            raise KeyError('Invalid Convert Type')

        self.dis_type = dis_type
        if dis_type == 'Gmm':
            self.dll = GmmDistribution(frame_size=num_features, k=k, multi_var=multi_var,
                                       differ_muk=differ_muk, muk_func=muk_func)
        elif dis_type == 'MultiGaussian':
            self.dll = MultiGaussian(frame_size=num_features, n_elements=n_elements)

        elif dis_type == 'Gaussian':
            self.dll = Gaussian(frame_size=50, n_elements=n_elements)

        else:
            self.dll = None
        self.EdgeLayer = EdgeLayer(dim=num_features, mlp_before=mlp_in_edge, n_channels=n_channels)

    def forward(self, inputs):
        # inputs: B C T
        inputs = self.convert_layer(inputs)
        outputs = inputs.transpose(-1, -2)
        # print('OUTPUTS', outputs.shape)
        edge = self.EdgeLayer(outputs)
        if not self.dll is None:
            params = self.dll(outputs)
        else:
            params = 0.0
        return outputs, edge, params