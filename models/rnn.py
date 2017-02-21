# PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from blocks import Linear, strnn, reverse_tensor


class Model(nn.Module):
    def __init__(self, inp_dim=None, mlp_dim=None, model_dim=None, num_classes=None, dropout_rate=0.5, **kwargs):
        super(Model, self).__init__()
        self.reverse = False
        self.bidirectional = False
        self.bi = 2 if self.bidirectional else 1
        self.num_layers = 1
        self.model_dim = model_dim

        self.rnn = nn.GRU(inp_dim, model_dim / self.bi, num_layers=self.num_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
            )

        self.l0 = nn.Linear(model_dim, mlp_dim)
        self.l1 = nn.Linear(mlp_dim, mlp_dim)
        self.l2 = nn.Linear(mlp_dim, num_classes)
        self.dropout_rate = dropout_rate

    def run_rnn(self, x):
        bi = self.bi

        batch_size, seq_len = x.size()[:2]
        model_dim = self.model_dim

        if self.reverse:
            x = reverse_tensor(x, dim=1)

        num_layers = self.num_layers
        h0 = Variable(torch.zeros(num_layers * bi, batch_size, model_dim / bi), volatile=not self.training)
        output, hn = self.rnn(x, h0)

        if self.reverse:
            output = reverse_tensor(output, dim=1)

        return output, hn

    def forward(self, x):
        _, x = self.run_rnn(x)
        x = x.squeeze() # TODO: This won't work for multiple layers or bidirectional.
        x = F.relu(F.dropout(self.l0(x), self.dropout_rate, self.training))
        x = F.relu(F.dropout(self.l1(x), self.dropout_rate, self.training))
        x = self.l2(x)
        return x
