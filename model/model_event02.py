"""Model 435 for Event detection using unet1d."""

from functools import partial
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import Dnet
from layers import gaussian_kernel
from layers import KLLoss
from layers import makeConvLayer
from layers import makeConvLayer1D
from layers import ResBlock
from layers import Sgate
from layers import Unet
from layers import Unet1d

import img_aug
from util import transform_label_kl


kernel_size = 31
kernel_std = 5
kernel = gaussian_kernel(kernel_size, kernel_std).view(1, 1, -1)
kernel.requires_grad = False

criterion = KLLoss()
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
    makeConvLayer(num_channel,16, 7, 2, 0),
    Sgate(),
    Unet(3, 16, 32),
    nn.MaxPool2d(2,2),
    makeConvLayer(16, 3, 1, 1, 0),
    Sgate(),
    )

  def forward(self, x):
    return self.model(x)


class Net(nn.Module):
  def __init__(self, out_dim, dropout, num_input_channel):
    super(Net, self).__init__()
    self.conv_net = ConvNet(num_input_channel)
    self.linear0 = nn.Linear(12 * 12 * 3, 256)
    self.dropout = nn.Dropout(dropout)
    self.linear1 = nn.Linear(256, out_dim)

    self.unet1d_moddule = nn.Sequential(
        Unet1d(3, 256, 512),
        makeConvLayer1D(256, 256, 1, 1, 0),
        Unet1d(3, 256, 512),
        )

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

    if seq_len % 16 != 0:
      padding = math.ceil(seq_len / 16) * 16 - seq_len
      out = F.pad(out, (0, padding), 'constant', 0)

    out = self.unet1d_moddule(out)

    if seq_len % 16 != 0:
      out = out[:, :, :seq_len]

    out = self.dropout(out)
    out = out.permute(0, 2, 1)
    out = out.reshape(batch_size * seq_len, -1)
    out = self.linear1(out)
    out = out.reshape(batch_size, seq_len, -1)
    out = torch.softmax(out, dim=2)

    return out
