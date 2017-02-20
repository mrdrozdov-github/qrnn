# PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class Linear(nn.Linear):

    def forward(self, x):
        shape = x.size()
        if len(shape) == 3:
            x = x.view(-1, shape[2])
        y = super(Linear, self).forward(x)
        if len(shape) == 3:
            y = y.view(shape[0], shape[1], y.size(1))
        return y


def strnn(f, z, hinit):
    batch_size, seq_length, model_dim = f.size()
    h = [hinit]

    # Kernel
    for t in range(seq_length):
        prev_h = h[-1]
        ft = f[:, t, :]
        zt = z[:, t, :]

        ht = prev_h * ft + zt
        h.append(ht)

    hs = torch.cat([hh.unsqueeze(1) for hh in h[1:]], 1)
    return hs