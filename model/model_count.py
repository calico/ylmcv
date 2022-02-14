import math
import sys

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import img_aug
from layers import Dnet
from layers import makeConvLayer
from layers import makeConvLayer1D
from layers import ResBlock
from layers import Sgate


criterion = nn.L1Loss()
sample_size=[1, 53, 53]

def jitter_func(src_width, src_height, out_width, out_height):
  return img_aug.getImageAugMat(src_width, src_height, out_width, out_height,
                               0.85, 1.0, 0.85, 1.0, True, math.pi / 20)


class ConvNet(nn.Module):
  def __init__(self, num_channel):
    super(ConvNet, self).__init__()
    self.block1 = nn.Sequential(
      makeConvLayer(num_channel, 16, 7, 2, 0),
      Sgate(),
      makeConvLayer(16, 48, 3, 1, 1),
      Sgate(),
      )

    self.resblock = nn.Sequential(
        ResBlock(48, 0.0),
        ResBlock(48, 0.0),
        nn.MaxPool2d(2, 2),
        makeConvLayer(48, 96, 1, 1, 0),
        Sgate(),
        ResBlock(96, 0.0),
        ResBlock(96, 0.0),
        ResBlock(96, 0.0),
        nn.MaxPool2d(2, 2),
        makeConvLayer(96, 192, 1, 1, 0),
        Sgate(),
        ResBlock(192, 0.0),
        ResBlock(192, 0.0),
        ResBlock(192, 0.0),
        nn.MaxPool2d(2, 2),
        makeConvLayer(192, 384, 1, 1, 0),
        Sgate(),
        ResBlock(384, 0.0),
        ResBlock(384, 0.0),
        ResBlock(384, 0.0),
        Sgate(),
        )

  def out_dim(self):
    return 384 * 3 * 3

  def forward(self, x):
    out1 = self.block1(x)
    out = self.resblock(out1)
    return out


class Net(nn.Module):
  def __init__(self, out_dim, dropout, num_input_channel):
    super(Net, self).__init__()
    self.conv_net = ConvNet(num_input_channel)
    input_dim = self.conv_net.out_dim()
    self.linear0 = nn.Linear(input_dim, 256)
    self.input_dim = 256
    self.out_dim = out_dim
    self.dropout = nn.Dropout(dropout)
    self.linear1 = nn.Linear(256, out_dim)

    self.dnet_module = nn.Sequential(
      Dnet(4, 256, 256),
      makeConvLayer1D(256, 256, 1, 1, 0),
      Sgate(),
      Dnet(4, 256, 256),
      makeConvLayer1D(256, 256, 1, 1, 0),
      Sgate(),
      )

  def forward(self, inputs):
    #convolutional output
    batch_size, seq_len, num_channel, h, w = inputs.shape
    out = inputs.view(-1, num_channel, h, w)
    out = self.conv_net(out)
    out = out.reshape(batch_size * seq_len, -1)
    out = self.linear0(out)
    out = out * torch.sigmoid(out)
    out = out.reshape(batch_size, seq_len, -1)
    out = out.permute(0, 2, 1)

    out = self.dnet_module(out)
    out = self.dropout(out)
    out = out.permute(0, 2, 1)
    out = out.reshape(-1, 256)
    out = self.linear1(out)

    out = out.reshape(batch_size, seq_len)
    out = torch.sigmoid(out)
    out = out.cumsum(dim=1)
    return out

