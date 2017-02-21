# PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from blocks import Linear, strnn, reverse_tensor


class Model(nn.Module):
    def __init__(self, inp_dim=None, mlp_dim=None, model_dim=None, num_classes=None, dropout_rate=0.5, **kwargs):
        super(Model, self).__init__()
        self.projection = Linear(inp_dim, model_dim)
        self.l0 = nn.Linear(model_dim, mlp_dim)
        self.l1 = nn.Linear(mlp_dim, mlp_dim)
        self.l2 = nn.Linear(mlp_dim, num_classes)
        self.dropout_rate = dropout_rate

    def forward(self, x):
        x = self.projection(x)
        x = torch.sum(x, 1).squeeze()
        x = F.relu(F.dropout(self.l0(x), self.dropout_rate, self.training))
        x = F.relu(F.dropout(self.l1(x), self.dropout_rate, self.training))
        x = self.l2(x)
        return x
