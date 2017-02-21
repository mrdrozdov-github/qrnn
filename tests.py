import unittest

# PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

import models.cbow
import models.rnn
import models.qrnn
import models.rnn_pool


def default_args(**kwargs):
    args = dict()
    args['inp_dim'] = 50
    args['model_dim'] = 30
    args['mlp_dim'] = 64
    args['num_classes'] = 3
    args['kernel_size'] = 3
    for k, v in kwargs.iteritems():
        args[k] = v
    return args


def model_suite(model_cls, **kwargs):
    args = default_args(**kwargs)
    model = model_cls(**args)

    batch_size = 8
    seq_length = 10
    inp_dim = args['inp_dim']
    x = Variable(torch.FloatTensor(batch_size, seq_length, inp_dim))

    outp = model(x)


class ModelsTestCase(unittest.TestCase):

    def test_cbow(self):
        model_suite(models.cbow.Model)

    def test_rnn(self):
        model_suite(models.rnn.Model)

    def test_qrnn(self):
        model_suite(models.qrnn.Model, kernel_size=1)
        model_suite(models.qrnn.Model, kernel_size=2)
        model_suite(models.qrnn.Model, kernel_size=3)

    def test_rnn_pool(self):
        model_suite(models.rnn_pool.Model)


if __name__ == '__main__':
    unittest.main()
