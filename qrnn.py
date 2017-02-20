# Based off code from this blog post:
# https://metamind.io/research/new-neural-network-building-block-allows-faster-and-more-accurate-text-understanding/

# PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np


"""

TODO:

- [x] Kernel Size 1
- [x] Kernel Size 2
- [ ] Kernel Size N
- [ ] Attention
- [ ] Decoder
- [ ] GPU Support

"""


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
            attention=False,
            decoder=False,
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


def strnn(f, z, hinit):
    batch_size, seq_length, model_dim = f.size()
    h = [hinit]

    # Kernel
    for t in range(seq_length):
        prev_h = h[-1]
        ft = f[:, t, :]
        zt = f[:, t, :]

        ht = prev_h * ft + zt
        h.append(ht)

    hs = torch.cat([hh.unsqueeze(1) for hh in h[1:]], 1)
    return hs


def attention_sum(encoding, query):
    alpha = F.softmax(F.batch_matmul(encoding, query, transb=True))
    alpha, encoding = F.broadcast(alpha[:, :, :, None],
                                  encoding[:, :, None, :])
    return torch.sum(alpha * encoding, 1)


class Linear(nn.Linear):

    def forward(self, x):
        shape = x.size()
        if len(shape) == 3:
            x = x.view(-1, shape[2])
        y = super(Linear, self).forward(x)
        if len(shape) == 3:
            y = y.view(shape[0], shape[1], y.size(1))
        return y


class QRNNLayer(nn.Module):

    def __init__(self, in_size, out_size, kernel_size=2, attention=False,
                 decoder=False):
        super(QRNNLayer, self).__init__()
        if kernel_size == 1:
            self.W = Linear(in_size, 3 * out_size)
        elif kernel_size == 2:
            self.W = Linear(in_size, 3 * out_size, bias=False)
            self.V = Linear(in_size, 3 * out_size)
        else:
            self.conv = L.ConvolutionND(1, in_size, 3 * out_size, kernel_size,
                                     stride=1, pad=kernel_size - 1)
        if attention:
            self.U = Linear(out_size, 3 * in_size)
            self.o = Linear(2 * out_size, out_size)
        self.in_size, self.size, self.attention = in_size, out_size, attention
        self.kernel_size = kernel_size

    def pre(self, x):
        dims = len(x.size()) - 1

        if self.kernel_size == 1:
            ret = self.W(x)
        elif self.kernel_size == 2:
            if dims == 2:
                xprev = Variable(torch.from_numpy(
                    np.zeros((self.batch_size, 1, self.in_size),
                                  dtype=np.float32)), volatile=not self.training)
                xtminus1 = torch.cat((xprev, x[:, :-1, :]), 1)
            else:
                xtminus1 = self.x
            ret = self.W(x) + self.V(xtminus1)
        else:
            ret = F.swapaxes(self.conv(
                F.swapaxes(x, 1, 2))[:, :, :x.size()[2]], 1, 2)

        if not self.attention:
            return ret

        if dims == 1:
            enc = self.encoding[:, -1, :]
        else:
            enc = self.encoding[:, -1:, :]
        return sum(F.broadcast(self.U(enc), ret))

    def init(self, encoder_c=None, encoder_h=None):
        self.encoding = encoder_c
        self.c, self.x = None, None
        if self.encoding is not None:
            self.batch_size = self.encoding.size()[0]
            if not self.attention:
                self.c = self.encoding[:, -1, :]

        if self.c is None or self.c.size()[0] < self.batch_size:
            self.c = Variable(torch.from_numpy(np.zeros((self.batch_size, self.size),
                                            dtype=np.float32)), volatile=not self.training)

        if self.x is None or self.x.size()[0] < self.batch_size:
            self.x = Variable(torch.from_numpy(np.zeros((self.batch_size, self.in_size),
                                            dtype=np.float32)), volatile=not self.training)

    def forward(self, x):
        if not hasattr(self, 'encoding') or self.encoding is None:
            self.batch_size = x.size()[0]
            self.init()
        dims = len(x.size()) - 1
        f, z, o = torch.chunk(self.pre(x), 3, dims)
        f = F.sigmoid(f)
        z = (1 - f) * F.tanh(z)
        o = F.sigmoid(o)

        if dims == 2:
            self.c = strnn(f, z, self.c[:self.batch_size])
        else:
            self.c = f * self.c + z

        if self.attention:
            context = attention_sum(self.encoding, self.c)
            self.h = o * self.o(torch.cat((self.c, context), dims))
        else:
            # self.c.unsqueeze(1).expand_as(o) <= broadcasting hack
            self.h = self.c * o

        self.x = x
        return self.h

    def get_state(self):
        return torch.cat((self.x, self.c, self.h), 1)

    def set_state(self, state):
        self.x, self.c, self.h = torch.chunk(
            state, (self.in_size, self.in_size + self.size), 1)

    state = property(get_state, set_state)
