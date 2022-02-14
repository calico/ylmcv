"""Various layers and modules."""


import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_kernel(size, stdv):
    x = torch.arange(1, size+1).float()
    x.add_(-math.ceil(size / 2))
    x.mul_(x)
    x = x.div_(-2 * stdv * stdv)
    x.exp_()
    x.div_(math.sqrt(2 * math.pi) * stdv)
    x.div_(x.sum())
    return x


class KLLoss(nn.Module):
  """loss function for budding event detection."""
  def __init__(self, size_average=True):
    super(KLLoss, self).__init__()
    self.size_average = size_average

  def forward(self, input, target):
    effective_size = target.data.sum()
    if effective_size < 1:
      effective_size = 1

    loss = -(torch.log(input + 1e-12) * target).sum() / effective_size
    return loss


def makeConvLayer(f1, f2, kw, dw, pw):
  """Simple helper function to create a 2d convolution layer."""
  layer  = nn.Conv2d(f1, f2, (kw, kw), (dw, dw), (pw, pw))
  nl = kw * kw * f1
  layer.weight.data.normal_(0, math.sqrt(2. / nl))
  layer.bias.data.zero_()
  return layer


def makeConvLayer1D(f1, f2, kw, dw, pw, dilation=1):
  layer  = nn.Conv1d(f1, f2, kw, dw, pw, dilation=dilation)
  nl = kw * f1
  layer.weight.data.normal_(0, math.sqrt(2. / nl))
  layer.bias.data.zero_()
  return layer


class ResBlock_BottleNeck(nn.Module):
  def __init__(self, fin, drop=0.3):
    super(ResBlock_BottleNeck, self).__init__()
    self.conv1 = makeConvLayer(fin, fin // 2, 3, 1, 1),
    self.conv2 = makeConvLayer(fin // 2, fin, 3, 1, 1),
    self.dropout = nn.Dropout(drop)

  def forward(self, x):
    out = F.leaky_relu(x, 0.1)
    out = self.conv1(out)
    out = F.leaky_relu(out, 0.1)
    out = self.conv2(out)
    return out + self.dropout(x)


class ResBlock(nn.Module):
  def __init__(self, fin, drop=0.15):
    super(ResBlock, self).__init__()
    self.conv1 = makeConvLayer(fin, fin // 2, 3, 1, 1)
    self.conv2 = makeConvLayer(fin // 2, fin, 3, 1, 1)
    self.dropout = nn.Dropout(drop)

  def forward(self, x):
    out = F.leaky_relu(x, 0.1)
    out = self.conv1(out)
    out = F.leaky_relu(out, 0.1)
    out = self.conv2(out)
    return out + self.dropout(x)


class Concat(nn.Module):
  def __init__(self, m1, m2):
    super(Concat, self).__init__()
    self.m1 = m1
    self.m2 = m2

  def forward(self, x):
    x1 = self.m1.forward(x)
    x2 = self.m2.forward(x)
    return torch.cat([x1, x2], 1)


class Upsample(nn.Module):
  def __init__(self, scale_factor=2, mode="nearest"):
    super(Upsample, self).__init__()
    self.scale_factor = scale_factor
    self.mode = mode

  def forward(self, x):
    return nn.functional.interpolate(x,
        scale_factor=self.scale_factor, mode=self.mode,
        align_corners=True)


class Add(nn.Module):
  def __init__(self, m1, m2):
    super(Add, self).__init__()
    self.m1 = m1
    self.m2 = m2

  def forward(self, x):
    x1 = self.m1.forward(x)
    x2 = self.m2.forward(x)
    return x1+x2


class Upsample1d(nn.Module):
  def __init__(self):
    super(Upsample1d, self).__init__()

  def forward(self, x):
    sz = list(x.size())
    x = x.reshape(*sz, 1)
    out = torch.cat([x,x], dim = 3)
    return out.reshape(sz[0], sz[1], -1)


def Unet(n, inDim, nextDim, mode="bilinear", dropout=0.1):
  """Simple implementation of a Unet."""

  assert n > 0, "n must be greater than 0"
  module_list = [
    makeConvLayer(inDim, nextDim, 3, 1, 1),
    nn.LeakyReLU(0.1),
    makeConvLayer(nextDim, nextDim, 1, 1, 0),
    nn.LeakyReLU(0.1),
    makeConvLayer(nextDim, nextDim, 3, 1, 1),
    nn.LeakyReLU(0.1),
    ]

  next_branch_list = []
  next_branch_list.append(nn.MaxPool2d(2, 2))

  if n > 1:
    next_branch_list.append(Unet(n-1, nextDim, nextDim * 2))
  else:
    next_branch_list.append(makeConvLayer(nextDim, 2 * nextDim, 3, 1, 1))
    next_branch_list.append(nn.LeakyReLU(0.1))
    next_branch_list.append(makeConvLayer(2 * nextDim, 2 * nextDim, 1, 1, 0))
    next_branch_list.append(nn.LeakyReLU(0.1))
    next_branch_list.append(makeConvLayer(2 * nextDim, nextDim, 3, 1, 1))
    next_branch_list.append(nn.LeakyReLU(0.1))

  next_branch_list.append(Upsample(scale_factor=2, mode=mode))
  next_branch_list.append(nn.Dropout(dropout))

  next_branch = nn.Sequential(*next_branch_list)
  concat = Concat(nn.Dropout(dropout), next_branch)

  module_list.append(concat)
  module_list.append(makeConvLayer(nextDim * 2, nextDim, 3, 1, 1))
  module_list.append(nn.LeakyReLU(0.1))
  module_list.append(makeConvLayer(nextDim, inDim, 3, 1, 1))
  module_list.append(nn.LeakyReLU(0.1))

  return nn.Sequential(*module_list)


def Unet1d(n, inDim, nextDim):
  """1D unet."""

  assert n > 0, "n must be greater than 0"
  module_list = [
      makeConvLayer1D(inDim,nextDim,3,1,1),
      nn.LeakyReLU(0.1),
      makeConvLayer1D(nextDim,inDim,1,1,0),
      nn.LeakyReLU(0.1),
      ]

  next_branch_list = [nn.MaxPool1d(2)]
  if n > 1:
    next_branch_list.append(Unet1d(n-1, inDim, nextDim))
  else:
    next_branch_list.append(makeConvLayer1D(inDim, nextDim, 3, 1, 1))
    next_branch_list.append(nn.LeakyReLU(0.1))
    next_branch_list.append(makeConvLayer1D(nextDim, inDim, 1, 1, 0))
    next_branch_list.append(nn.LeakyReLU(0.1))

  next_branch_list.append(Upsample1d())
  next_branch_list.append(nn.Dropout(0.1))
  next_branch = nn.Sequential(*next_branch_list)

  module_list.append(Add(nn.Dropout(0.1), next_branch))
  module_list.append(makeConvLayer1D(inDim, nextDim, 1, 1, 0))
  module_list.append(nn.LeakyReLU(0.1))
  module_list.append(makeConvLayer1D(nextDim,inDim, 3, 1, 1))
  module_list.append(nn.LeakyReLU(0.1))

  return nn.Sequential(*module_list)


class DilateModule(nn.Module):
  """A simple module for dilated convolution with gating."""

  def __init__(self, f1, f2, kw, dw, pw, diw, dropout=0):
    super(DilateModule, self).__init__()

    if type(pw) == tuple:
      self.conv1 = makeConvLayer1D(f1, 2 * f2, kw, dw, 0, dilation=diw)
      self.pad = True
      self.left_pad, self.right_pad = pw
    else:
      self.pad = False
      self.conv1 = makeConvLayer1D(f1, 2 * f2, kw, dw, pw, dilation=diw)

    self.f1 = f1
    self.f2 = f2
    self.dropout = nn.Dropout(dropout)
    if f1 != f2:
      self.conv2 = makeConvLayer1D(f1, f2, 1, 1, 0)

  def forward(self, x):
    if self.pad:
      out = F.pad(x, (self.left_pad, self.right_pad), 'constant', 0)
      conv1_out = self.conv1(out)
    else:
      conv1_out = self.conv1(x)
    out1 = torch.sigmoid(conv1_out.narrow(1, 0, self.f2))
    out2 = torch.tanh(conv1_out.narrow(1, self.f2, self.f2))
    out = out1 * out2
    if self.f1 == self.f2:
      out = self.dropout(x) + out
    else:
      out1 = self.conv2(x)
      out1 = F.leaky_relu(out1, 0.1)
      out = self.dropout(out1) + out
    return out


class Padding(nn.Module):
  def __init__(self, pad):
    super(Padding, self).__init__()
    if type(pad) == tuple:
      self.left_pad, self.right_pad = pad
    else:
      self.left_pad = pad
      self.right_pad = pad

  def forward(self, x):
    out = F.pad(x, (self.left_pad, self.right_pad), 'constant', 0)
    return out


def Dnet(n, inDim, nextDim, causal=False):
  """Dilated convolution network."""

  assert n > 0, "n must be greater than 0"
  module_list = []
  for i in range(n-1):
    if causal:
      module_list.append(DilateModule(inDim, nextDim, 3, 1,
        (2 * 2 ** i, 0), 2 ** i))
    else:
      module_list.append(DilateModule(inDim, nextDim, 3, 1, 2 ** i, 2 ** i))
  return nn.Sequential(*module_list)


class Sgate(nn.Module):
  """sgate activation."""
  def __init__(self):
    super(Sgate, self).__init__()

  def forward(self, x):
    return x * torch.sigmoid(x)
