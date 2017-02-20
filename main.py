import gflags
import time
import sys

import numpy as np

# PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

from torchtext import data
from torchtext import datasets

from collections import OrderedDict

from qrnn import QRNNModel

FLAGS = gflags.FLAGS


class CBOW(nn.Module):
    def __init__(self, inp_dim=None, mlp_dim=None, num_classes=None, dropout_rate=0.5, **kwargs):
        super(CBOW, self).__init__()
        self.l0 = nn.Linear(inp_dim, mlp_dim)
        self.l1 = nn.Linear(mlp_dim, mlp_dim)
        self.l2 = nn.Linear(mlp_dim, num_classes)
        self.dropout_rate = dropout_rate

    def forward(self, x):
        x = torch.sum(x, 1).squeeze()
        x = F.relu(F.dropout(self.l0(x), self.dropout_rate, self.training))
        x = F.relu(F.dropout(self.l1(x), self.dropout_rate, self.training))
        x = self.l2(x)
        return x


class RNN(nn.Module):
    def __init__(self, inp_dim=None, mlp_dim=None, model_dim=None, num_classes=None, dropout_rate=0.5, **kwargs):
        super(RNN, self).__init__()
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


def reverse_tensor(var, dim):
    dim_size = var.size(dim)
    index = [i for i in range(dim_size - 1, -1, -1)]
    index = torch.LongTensor(index)
    if isinstance(var, Variable):
        index = to_gpu(Variable(index, volatile=var.volatile))
    inverted_tensor = var.index_select(dim, index)
    return inverted_tensor


def get_output(model, batch, embed, train=False):
    if train:
        model.train()
    else:
        model.eval()

    # Build input.
    x = batch.text.t() # reshape to (B, S)
    batch_size, seq_length = x.size()
    x = embed(x.contiguous().view(-1)) # embed
    x = x.view(batch_size, seq_length, -1) # reshape to (B, S, E)
    x = Variable(x.data, volatile=not train) # break the computational chain

    # Calculate loss and update parameters.
    outp = model(x)

    return outp


def run():

    # From torchtext source:
    # set up fields
    TEXT = data.Field()
    LABEL = data.Field(sequential=False)

    # make splits for data
    train, val = datasets.SST.splits(
        TEXT, LABEL, fine_grained=False, train_subtrees=False,
        test=None, train='train.txt' if not FLAGS.demo else 'dev.txt',
        filter_pred=lambda ex: ex.label != 'neutral')

    # print information about the data
    print('train.fields', train.fields)
    print('len(train)', len(train))
    print('vars(train[0])', vars(train[0]))

    # build the vocabulary
    TEXT.build_vocab(train, wv_type=FLAGS.wv_type, wv_dim=FLAGS.wv_dim)
    LABEL.build_vocab(train)

    # print vocab information
    print('len(TEXT.vocab)', len(TEXT.vocab))
    print('TEXT.vocab.vectors.size()', TEXT.vocab.vectors.size())

    # make iterator for splits
    train_iter, val_iter = data.BucketIterator.splits(
        (train, val), batch_size=FLAGS.batch_size, device=FLAGS.gpu)

    # Build model.
    if FLAGS.model_type == "cbow":
        model_cls = CBOW
    elif FLAGS.model_type == "rnn":
        model_cls = RNN
    elif FLAGS.model_type == "qrnn":
        model_cls = QRNNModel
    else:
        raise NotImplementedError
    model = model_cls(
        inp_dim=FLAGS.wv_dim,
        model_dim=FLAGS.model_dim,
        mlp_dim=FLAGS.mlp_dim,
        num_classes=FLAGS.num_classes,
        kernel_size=FLAGS.kernel_size,
        )

    # Build optimizer.
    optimizer = optim.Adam(model.parameters())

    print(model)
    total_params = sum([reduce(lambda x, y: x * y, p.size()) for p in model.parameters()])
    print(total_params)

    # Pre-trained embedding layer.
    embed = nn.Embedding(TEXT.vocab.vectors.size(0), TEXT.vocab.vectors.size(1))
    embed.load_state_dict(OrderedDict([('weight', TEXT.vocab.vectors)]))

    # Stats.
    trailing_acc = 0.0
    trailing_time = 0.0

    step = 0
    epoch = 0

    while True:
        # Main train loop.
        for batch_idx, batch in enumerate(train_iter):
            start = time.time()

            # Build target.
            y = batch.label

            outp = get_output(model, batch, embed, train=True)
            dist = F.log_softmax(outp)
            loss = nn.NLLLoss()(dist, y)

            model.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate accuracy.
            preds = dist.data.max(1)[1]
            acc = y.data.eq(preds).sum() / float(y.size(0))
            trailing_acc = 0.9 * trailing_acc + 0.1 * acc

            end = time.time()

            # Calculate time per token.
            num_tokens = reduce(lambda x, y: x * y, batch.text.size())
            time_per_token = (end-start) / float(num_tokens)
            trailing_time = 0.9 * trailing_time + 0.1 * time_per_token

            # Periodically print statistics.
            if step % FLAGS.statistics_interval_steps == 0:
                print("Step: {} [{}] Loss: {} Acc: {} Time: {:.10f}".format(step, epoch, loss.data[0], trailing_acc, trailing_time))

            if step > 0 and step % FLAGS.eval_interval_steps == 0:
                start = time.time()
                total_tokens = 0
                total_correct = 0
                total = 0
                for eval_batch_idx, eval_batch in enumerate(val_iter):
                    y = eval_batch.label
                    outp = get_output(model, eval_batch, embed, train=False)
                    dist = F.log_softmax(outp)
                    preds = dist.data.max(1)[1]

                    total_tokens += reduce(lambda x, y: x * y, eval_batch.text.size())
                    total_correct += y.data.eq(preds).sum()
                    total += y.size(0)
                end = time.time()
                time_per_token = (end-start) / float(total_tokens)
                acc = total_correct / float(total)
                print("Eval Step: {} [{}] Acc: {} Time: {:.10f}".format(step, epoch, acc, time_per_token))

            step += 1
            if step > FLAGS.training_steps:
                quit()
        epoch += 1

if __name__ == '__main__':
    # Debug settings.
    gflags.DEFINE_boolean("demo", False, "Set to True to use dev data for training, which will load faster.")

    # Device settings.
    gflags.DEFINE_integer("gpu", -1, "")

    # Data settings.
    gflags.DEFINE_integer("batch_size", 8, "")
    gflags.DEFINE_string("wv_type", "glove.6B", "")
    gflags.DEFINE_integer("wv_dim", 50, "")

    # Model settings.
    gflags.DEFINE_enum("model_type", "qrnn", ["cbow", "rnn", "rnn-gate", "qrnn"], "")
    gflags.DEFINE_integer("kernel_size", 3, "")
    gflags.DEFINE_integer("model_dim", 100, "")
    gflags.DEFINE_integer("mlp_dim", 256, "")
    gflags.DEFINE_integer("num_classes", 3, "")

    # Train settings.
    gflags.DEFINE_integer("training_steps", 10000, "")

    # Log settings.
    gflags.DEFINE_integer("statistics_interval_steps", 100, "")
    gflags.DEFINE_integer("eval_interval_steps", 100, "")

    # Read command line options.
    FLAGS(sys.argv)

    run()
