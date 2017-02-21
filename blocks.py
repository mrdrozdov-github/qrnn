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


def reverse_tensor(var, dim):
    dim_size = var.size(dim)
    index = [i for i in range(dim_size - 1, -1, -1)]
    index = torch.LongTensor(index)
    if isinstance(var, Variable):
        index = to_gpu(Variable(index, volatile=var.volatile))
    inverted_tensor = var.index_select(dim, index)
    return inverted_tensor
