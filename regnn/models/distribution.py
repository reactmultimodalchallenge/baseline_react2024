import torch
import torch.nn as nn
import math

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

class GmmDistribution(nn.Module):
    def __init__(self, frame_size, k=6, multi_var=False, differ_muk=False, muk_func=False):
        super().__init__()
        self.k = k
        self.frame_size = frame_size
        self.multi_var = multi_var
        self.differ_muk = differ_muk
        self.muk_func = muk_func
        for i in range(k):
            self.__setattr__(name='mean_' + str(i), value=Mlp(in_features=frame_size))
            if multi_var:
                self.__setattr__(name='var_' + str(i), value=Mlp(in_features=frame_size))

        if differ_muk:
            if not muk_func:
                self.muk = nn.Parameter(torch.ones((k, 1), requires_grad=True))
            else:
                self.muk_func = nn.Sequential(
                    Mlp(in_features=frame_size, hidden_features=frame_size // 2, out_features=1),
                    nn.Softmax(dim=-1)
                                              )

    def forward(self, inputs):
        means = []
        if self.multi_var:
            vars = []

        for i in range(self.k):
            mean_i_func = self.__getattr__(name='mean_' + str(i))
            mean_i = mean_i_func(inputs)
            means.append(mean_i)
            if self.multi_var:
                var_i_func = self.__getattr__(name='var_' + str(i))
                var_i = var_i_func(inputs)
                vars.append(var_i)

        mean = torch.stack(means, dim=0) # k 25 frame_size
        if self.multi_var:
            var = torch.stack(vars, dim=0)
        else:
            var = torch.ones((self.k, 25, self.frame_size))

        if self.differ_muk:
            if self.muk_func:
                muk = self.muk_func(mean)
            else:
                muk = self.muk / self.muk.sum()
        else:
            muk = 1 / self.k

        return mean, var, muk

class MultiGaussian(nn.Module):
    def __init__(self, frame_size, n_elements):
        super().__init__()
        self.frame_size = frame_size
        self.n_elements = n_elements
        for i in range(n_elements):
            self.__setattr__(name='fc_' + str(i), value=Mlp(in_features=frame_size))

        self.var_func = Mlp(in_features=2 * n_elements, hidden_features=n_elements, out_features=1)

    def forward(self, inputs):
        # inputs: (B, 25, frame_size)
        means = []
        for i in range(self.n_elements):
            fc_i = self.__getattr__(name='fc_' + str(i))
            mean_i = fc_i(inputs[i] if len(inputs.shape) == 2 else inputs[:, i])
            means.append(mean_i)

        means = torch.cat(means, dim=0)

        var = self.var_func(torch.cat((inputs.squeeze(0) if len(inputs.shape) == 3 else inputs,
                                       means), dim=0).transpose(-2, -1)
                            )

        # means: (25, frame_size), var: (1, frame_size)
        return means, var

class Gaussian(nn.Module):
    def __init__(self, frame_size, n_elements):
        super().__init__()
        self.frame_size = frame_size
        self.n_elements = n_elements
        self.mean_func =Mlp(in_features=n_elements * frame_size)

        self.var_func = Mlp(in_features=n_elements * frame_size)

    def forward(self, inputs):
        # inputs: (B, 25, frame_size)
        inputs = inputs.reshape(-1, self.n_elements * self.frame_size)
        mean = self.mean_func(inputs)
        var = self.var_func(inputs)
        var = torch.eye(self.n_elements * self.frame_size, device='cuda:0')
        # mean var : B 25*frame_size
        return mean, var