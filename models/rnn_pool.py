# PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from blocks import Linear, strnn

import numpy as np


class Model(nn.Module):
    def __init__(self, inp_dim=None, mlp_dim=None, model_dim=None, num_classes=None, dropout_rate=0.5, **kwargs):
        super(Model, self).__init__()
        self.model_dim = model_dim
        self.projection = Linear(inp_dim, model_dim * 3)
        self.l0 = nn.Linear(model_dim, mlp_dim)
        self.l1 = nn.Linear(mlp_dim, mlp_dim)
        self.l2 = nn.Linear(mlp_dim, num_classes)
        self.dropout_rate = dropout_rate

    def forward(self, x):
        x = self.projection(x)
        batch_size = x.size(0)

        f, z, o = torch.chunk(x, 3, 2)
        f = F.sigmoid(f)
        z = (1 - f) * F.tanh(z)
        o = F.sigmoid(o)

        c = Variable(torch.from_numpy(np.zeros((batch_size, self.model_dim),
                dtype=np.float32)), volatile=not self.training)

        c = strnn(f, z, c)
        h = c * o

        hn = h[:, -1, :]

        hn = F.relu(F.dropout(self.l0(hn), self.dropout_rate, self.training))
        hn = F.relu(F.dropout(self.l1(hn), self.dropout_rate, self.training))
        hn = self.l2(hn)
        return hn
