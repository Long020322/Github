import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, num_embed):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(num_embed, num_embed),
            nn.Tanh(),
            nn.Linear(num_embed, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1)


class ModeProduct(nn.Module):
    def __init__(self, i, j, k, a, b, c):
        super(ModeProduct, self).__init__()
        self.A = nn.Parameter(torch.rand(i, a))
        self.B = nn.Parameter(torch.rand(j, b))
        self.C = nn.Parameter(torch.rand(k, c))

        self.b = nn.Parameter(torch.rand(a, b, c))

    def forward(self, x):
        ret = torch.sigmoid(
            torch.einsum('nijk,kc->nijc', torch.einsum('nijk,jc->nick', torch.einsum('nijk,ic->ncjk', x, self.A), self.B), self.C) + self.b)
        return ret
