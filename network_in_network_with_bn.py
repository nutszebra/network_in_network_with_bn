import six
import numpy as np
import functools
import chainer.links as L
import chainer.functions as F
from collections import defaultdict
import nutszebra_chainer


class MlpBNReLUConv(nutszebra_chainer.Model):

    def __init__(self, in_channel, out_channels=(192, 160, 96), filter_sizes=(5, 1, 1), strides=(1, 1, 1), pads=(2, 0, 0)):
        super(MlpBNReLUConv, self).__init__()
        modules = []
        modules += [('bn1', L.BatchNormalization(in_channel))]
        modules += [('conv1', L.Convolution2D(in_channel, out_channels[0], filter_sizes[0], strides[0], pads[0]))]
        for i in six.moves.range(2, len(out_channels) + 1):
            modules += [('bn{}'.format(i), L.BatchNormalization(out_channels[i - 2]))]
            modules += [('conv{}'.format(i), L.Convolution2D(out_channels[i - 2], out_channels[i - 1], filter_sizes[i - 1], strides[i - 1], pads[i - 1]))]
        # register layers
        [self.add_link(*link) for link in modules]
        self.modules = modules
        self.in_channel = in_channel
        self.out_channels = out_channels
        self.filter_sizes = filter_sizes
        self.strides = strides
        self.pads = pads

    @staticmethod
    def _count_conv_parameters(conv):
        return functools.reduce(lambda a, b: a * b, conv.W.data.shape)

    def count_parameters(self):
        count = 0
        for name, link in self.modules:
            if 'bn' in name:
                continue
            count += MlpBNReLUConv._count_conv_parameters(link)
        return count

    def weight_initialization(self):
        for name, link in self.modules:
            if 'bn' in name:
                continue
            self[name].W.data = self.weight_relu_initialization(link)
            self[name].b.data = self.bias_initialization(link, constant=0)

    def __call__(self, x, train=False):
        for i in six.moves.range(1, len(self.out_channels) + 1):
            x = self['conv{}'.format(i)](F.relu(self['bn{}'.format(i)](x, test=not train)))
        return x


class Network_In_Network(nutszebra_chainer.Model):

    def __init__(self, category_num):
        super(Network_In_Network, self).__init__()
        modules = []
        modules += [('mlpconv1', MlpBNReLUConv(3, (192, 160, 96), (5, 1, 1), (1, 1, 1), (2, 0, 0)))]
        modules += [('mlpconv2', MlpBNReLUConv(96, (192, 192, 192), (5, 1, 1), (1, 1, 1), (2, 0, 0)))]
        modules += [('mlpconv3', MlpBNReLUConv(192, (192, 192, category_num), (3, 1, 1), (1, 1, 1), (1, 0, 0)))]
        # register layers
        [self.add_link(*link) for link in modules]
        self.modules = modules
        self.category_num = category_num
        self.name = 'Network_In_Network_{}'.format(category_num)

    def count_parameters(self):
        count = 0
        for name, link in self.modules:
            count += link.count_parameters()
        return count

    def weight_initialization(self):
        for name, link in self.modules:
            link.weight_initialization()

    def __call__(self, x, train=False):
        h = self.mlpconv1(x, train=train)
        h = F.max_pooling_2d(h, ksize=(3, 3), stride=(2, 2), pad=(0, 0))
        h = F.dropout(h, ratio=0.5, train=train)
        h = self.mlpconv2(h, train=train)
        h = F.average_pooling_2d(h, ksize=(3, 3), stride=(2, 2), pad=(0, 0))
        h = F.dropout(h, ratio=0.5, train=train)
        h = self.mlpconv3(h, train=train)
        num, categories, y, x = h.data.shape
        # global average pooling
        h = F.reshape(F.average_pooling_2d(h, (y, x)), (num, categories))
        return h

    def calc_loss(self, y, t):
        loss = F.softmax_cross_entropy(y, t)
        return loss

    def accuracy(self, y, t, xp=np):
        y.to_cpu()
        t.to_cpu()
        indices = np.where((t.data == np.argmax(y.data, axis=1)) == True)[0]
        accuracy = defaultdict(int)
        for i in indices:
            accuracy[t.data[i]] += 1
        indices = np.where((t.data == np.argmax(y.data, axis=1)) == False)[0]
        false_accuracy = defaultdict(int)
        false_y = np.argmax(y.data, axis=1)
        for i in indices:
            false_accuracy[(t.data[i], false_y[i])] += 1
        return accuracy, false_accuracy
