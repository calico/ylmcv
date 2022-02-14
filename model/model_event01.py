from functools import partial
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

import img_aug
from layers import Dnet
from layers import gaussian_kernel
from layers import KLLoss
from layers import makeConvLayer
from layers import makeConvLayer1D
from layers import ResBlock
from layers import Sgate
from layers import Unet

from util import transform_label_kl


kernel_size = 51
kernel_std = 7
kernel = gaussian_kernel(kernel_size, kernel_std).view(1, 1, -1)
kernel.requires_grad = False

transform_label = partial(transform_label_kl, kernel=kernel,
    kernel_size=kernel_size)
label_size = 3
sample_size=[1, 53, 53]


def jitter_func(src_width, src_height, out_width, out_height):
  return img_aug.getImageAugMat(src_width, src_height, out_width, out_height,
                               0.85, 1.0, 0.85, 1.0, True, math.pi / 20)


class ConvNet(nn.Module):
  def __init__(self, num_channel=3):
    super(ConvNet, self).__init__()
    self.model = nn.Sequential(
    makeConvLayer(num_channel, 16, 7, 2, 0),
    Sgate(),
    Unet(3, 16, 32),
    nn.MaxPool2d(2, 2),
    makeConvLayer(16, 32, 1, 1, 0),
    Sgate(),
    makeConvLayer(32, 16, 3, 1, 1),
    Sgate(),
    makeConvLayer(16, 16, 3, 1, 1),
    Sgate(),
    makeConvLayer(16, 3, 3, 1, 1),
    Sgate(),
    )

  def forward(self, x):
    return self.model.forward(x)


class Net(nn.Module):
  def __init__(self, out_dim, dropout, num_input_channel):
    super(Net, self).__init__()
    self.conv_net = ConvNet(num_input_channel)
    self.linear0 = nn.Linear(12 * 12 * 3, 512)
    self.out_dim = out_dim
    self.dropout = nn.Dropout(dropout)

    self.dnet_module = nn.Sequential(
      Dnet(4, 512, 512),
      makeConvLayer1D(512, 1024, 1, 1, 0),
      Sgate(),
      Dnet(3, 1024, 1024),
      makeConvLayer1D(1024, 1024, 1, 1, 0),
      Sgate(),
      Dnet(4, 1024, 1024),
      makeConvLayer1D(1024, 1024, 1, 1, 0),
      Sgate(),
      )

    self.linear1 = nn.Linear(1024, out_dim)

  def forward(self, inputs):
    #convolutional output
    batch_size, seq_len, num_channel, h, w = inputs.shape
    out = inputs.reshape(-1, num_channel, h, w)
    out = self.conv_net(out)
    out = out.reshape(batch_size * seq_len, -1)
    out = self.linear0(out)
    out = out * torch.sigmoid(out)
    out = out.reshape(batch_size, seq_len, -1)
    out = out.permute(0, 2, 1)

    out = self.dnet_module(out)
    out = self.dropout(out)
    out = out.permute(0, 2, 1)
    out = out.reshape(batch_size * seq_len, -1)
    out = self.linear1(out)
    out = out.reshape(batch_size, seq_len, self.out_dim)
    out = F.softmax(out, dim=2)

    return out
