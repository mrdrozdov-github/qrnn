# Based off code from this blog post:
# https://metamind.io/research/new-neural-network-building-block-allows-faster-and-more-accurate-text-understanding/

# PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from blocks import Linear, strnn

import numpy as np


class QRNNModel(nn.Module):
    """docstring for QRNNModel"""
    def __init__(self, inp_dim=None, model_dim=None, mlp_dim=None, num_classes=None, dropout_rate=0.5,
                 kernel_size=None,
                 **kwargs):
        super(QRNNModel, self).__init__()
        self.qrnn = QRNNLayer(
            in_size=inp_dim,
            out_size=model_dim,
            kernel_size=kernel_size,
            )
        self.l0 = nn.Linear(model_dim, mlp_dim)
        self.l1 = nn.Linear(mlp_dim, mlp_dim)
        self.l2 = nn.Linear(mlp_dim, num_classes)
        self.dropout_rate = dropout_rate

    def forward(self, x):
        qs = self.qrnn(x)
        q = qs[:, -1, :]
        q = F.relu(F.dropout(self.l0(q), self.dropout_rate, self.training))
        q = F.relu(F.dropout(self.l1(q), self.dropout_rate, self.training))
        q = self.l2(q)
        return q


class QRNNLayer(nn.Module):

    def __init__(self, in_size, out_size, kernel_size=2):
        super(QRNNLayer, self).__init__()
        if kernel_size == 1:
            self.W = Linear(in_size, 3 * out_size)
        elif kernel_size == 2:
            self.W = Linear(in_size, 3 * out_size, bias=False)
            self.V = Linear(in_size, 3 * out_size)
        else:
            self.conv = nn.Conv1d(in_size, 3 * out_size, kernel_size,
                                     stride=1, padding=kernel_size - 1)
        self.in_size, self.size = in_size, out_size
        self.kernel_size = kernel_size

    def pre(self, x):
        if self.kernel_size == 1:
            ret = self.W(x)
        elif self.kernel_size == 2:
            xprev = Variable(torch.from_numpy(
                np.zeros((self.batch_size, 1, self.in_size),
                              dtype=np.float32)), volatile=not self.training)
            xtminus1 = torch.cat((xprev, x[:, :-1, :]), 1)
            ret = self.W(x) + self.V(xtminus1)
        else:
            ret = self.conv(x.transpose(1,2).contiguous()).transpose(1,2).contiguous()

        return ret

    def init(self, encoder_c=None, encoder_h=None):
        self.c = Variable(torch.from_numpy(np.zeros((self.batch_size, self.size),
                                            dtype=np.float32)), volatile=not self.training)

    def forward(self, x):
        self.batch_size = x.size()[0]
        self.init()

        dims = len(x.size()) - 1
        f, z, o = torch.chunk(self.pre(x), 3, dims)
        f = F.sigmoid(f)
        z = (1 - f) * F.tanh(z)
        o = F.sigmoid(o)

        self.c = strnn(f, z, self.c[:self.batch_size])
        self.h = self.c * o

        return self.h
