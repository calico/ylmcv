"""Utility functions."""
import math
import pickle
import random
import time

import numpy as np
import pandas as pd
import cv2

import torch
import torch.nn.functional as F


def warp_wrapper(aug_fn):
  def do_aug(image, target):
    """Input image should be of (n, h, w) or (n, h, w, c)."""
    if image.ndim == 3:
      image = np.expand_dims(image, axis=3)

    n, h, w, c = image.shape
    image = np.transpose(image, (1, 2, 0, 3))
    image = image.reshape(*image.shape[:2], -1)

    image, target = aug_fn(image, target)

    image = image.reshape(*image.shape[:2], n, c)
    image = np.transpose(image, (2, 3, 0, 1)) #nchw
    return image, target
  return do_aug


def GCN_wrapper(gcn):
  def do_gcn(image, target):
    """Input image should be of (n, c,  h, w)"""
    im_dim = image.ndim
    im_shape = image.shape
    if im_dim == 4:
      image = image.reshape(-1, *im_shape[2:])

    image, target = gcn(image, target)
    if im_dim == 4:
      image = image.reshape(*im_shape)

    return image, target
  return do_gcn


class RandChannel(object):
  """Randomly select a channel for multi-channel images."""

  def __init__(self, channel_idx=3):
    self.channel_idx = channel_idx

  def __call__ (self, data, target):
    if self.channel_idx < data.ndim:
      num_channel = data.shape[self.channel_idx]
      if num_channel > 1:
        idx = np.random.randint(num_channel, size=1)[0]
      data=np.take(data, 1, axis=self.channel_idx)
    return data, target


def transform_label(label, l2i, events=None):
  if events is not None:
    label.zero_()
    if len(events['b']) > 0:
      label[torch.LongTensor(events['b'])] = 1
    return label.cumsum(0).float()
  else:
    return (label == l2i["b"]).cumsum(0).float()


def transform_label_kl(label, l2i, kernel_size, kernel, events=None):
  assert events is not None, "only works for annotation with events"
  padding = kernel_size // 2
  label_new = torch.zeros(label.size(0) + padding, 3)

  # for events y, randomly assign it to x or w
  for e in events['y']:
    if np.random.rand(1)[0] > 0.5:
      events['x'].append(e)
    else:
      events['w'].append(e)

  tmp = torch.zeros(label.size(0) + padding)
  if len(events['x']) > 0:
    tmp[torch.LongTensor(events['x'])] = 1
    label_new[:, 0] =F.conv1d(tmp.view(1, 1, -1), kernel,
        padding=padding).data.view(-1)
    _max = label_new[:,0].max()
    if _max > 0:
      label_new[:, 0] /= _max

  tmp.zero_()
  if len(events['w']) > 0:
    tmp[torch.LongTensor(events['w'])] = 1
    label_new[:, 1] =F.conv1d(tmp.view(1, 1, -1), kernel,
        padding=padding).data.view(-1)
    _max = label_new[:, 1].max()
    if _max > 0:
      label_new[:, 1] /= _max

  _max = label_new[:, :2].sum(dim=1).max()
  if _max > 0:
    label_new[:, :2] /= _max
  label_new[:, 2] = 1 - label_new[:, :2].sum(dim=1)
  return label_new


class BudJitter(object):
  def __init__(self, random_init, frame_dropout):
    self.random_init = random_init
    self.frame_dropout = frame_dropout

  def __call__ (self, data, target):
    if target is None:
      return data, target

    data_len = data.shape[0]
    if self.random_init and data_len > 15:
      ss = np.random.randint(data_len // 3 * 2)
      if ss > 0:
        data = data[ss:]
        target = target[ss:]

    if self.frame_dropout > 0:
      r = torch.rand(2)
      if r[0] < self.frame_dropout:
        s = 1 if r[1] < 0.5 else 0
        idx1 = np.array(range(s, data.shape[0], 2))
        idx2 = np.array(range(s, target.shape[0], 2))
        data = data[idx1]
        target = target[idx2]

    return data, target


def read_meta(meta_file, offset):
  """Reading the meta data."""
  meta = pd.read_csv(meta_file)
  meta['end'] = meta['end'] - offset
  meta = meta[meta['end'] > 0]
  return {'id': meta['sample_id'].values, 'label': meta['end'].values}


def filter_list(alist, mask):
  return [alist[i] for i in range(len(mask)) if mask[i]]


def read_and_merge_meta(meta_files):
  meta_list = []
  for meta_file in meta_files.split(','):
    with open(meta_file, 'rb') as f:
      meta_list.append(pickle.load(f))
  if len(meta_list) == 1:
    return meta_list[0]

  meta = {}
  for k in ['label', 'id', 'events', 'symbols']:
    meta[k] = [x for m in meta_list for x in m[k]]
  meta['symbols'] = sorted(set(meta['symbols']))
  meta['symbols'].remove('$')
  meta['symbols'].append('$')
  meta['symbol2id'] = {v:i for i,v in enumerate(meta['symbols'])}
  for k, v in meta.items():
    print(k, len(v))
  return meta


def read_pkl_meta(meta_files, test_list_file, tfm_label, loss_type='L1',
    jpg_input_only=False):
  """Read the meta data in pickled format."""

  test_list = pd.read_csv(test_list_file, header=None).iloc[:,0]
  print ('number of test samples %s ' % len(test_list), flush=True)
  meta = read_and_merge_meta(meta_files)
  test_mask = pd.Series(meta['id']).isin(test_list).values
  train_mask = ~test_mask
  label2id = meta['symbol2id']
  id2label = meta['symbols']
  if jpg_input_only:
    meta['id'] = np.array([x.replace(".pkl", ".jpg") for x in meta['id']])

  cumsum=True

  if loss_type in ['NLL', 'BCE', 'KL', 'DUO']:
    cumsum=False

  meta['label'] = [tfm_label(torch.from_numpy(x).long(), label2id, events=y)
      for x, y in zip (meta['label'], meta['events'])]

  data_set_mask = np.array([not(x is None) for x in meta['label']], dtype=bool)
  print("number of samples.")
  print(len(meta['label']))
  print("number of good samples.")
  print(np.sum(data_set_mask))
  train_meta={'id':meta['id'][train_mask & data_set_mask],
      'label':filter_list(meta['label'], train_mask & data_set_mask)}
  test_meta={'id':meta['id'][test_mask & data_set_mask],
      'label':filter_list(meta['label'], test_mask & data_set_mask)}

  return train_meta, test_meta


def read_csv_meta(meta_file, offset, test_list_file, tfm_label, loss_type='L1'):
  """Read the meta data in csv format."""
  meta = pd.read_csv(meta_file)
  meta['end'] = meta['end'] - offset
  meta = meta[meta['end'] > 0]
  meta = {'id': meta['sample_id'].values, 'label': meta['end'].values}

  test_list = pd.read_csv(test_list_file, header=None).iloc[:, 0]
  print ('number of test samples %s ' % len(test_list), flush=True)
  test_mask = pd.Series(meta['id']).isin(test_list).values
  train_mask = ~test_mask

  meta['label'] = [tfm_label(int(x)) for x in meta['label']]

  data_set_mask = np.array([x is not None for x in meta['label']], dtype=bool)
  print("number of samples.")
  print(len(meta['label']))
  print("number of good samples.")
  print(np.sum(data_set_mask))
  train_meta={'id': meta['id'][train_mask & data_set_mask],
          'label': filter_list(meta['label'], train_mask & data_set_mask)}
  test_meta={'id': meta['id'][test_mask & data_set_mask],
          'label': filter_list(meta['label'], test_mask & data_set_mask)}

  return train_meta, test_meta


def slice_img(im, mid, left_offset, right_offset=None):
  if right_offset is None:
    right_offset = left_offset
  pad_shape  = list(im.shape)
  pad_shape[0] = left_offset
  left_zero_pad = np.zeros(pad_shape, dtype=im.dtype)
  pad_shape[0] = right_offset
  right_zero_pad = np.zeros(pad_shape, dtype=im.dtype)
  im = np.concatenate([left_zero_pad, im, right_zero_pad], axis=0)
  # because of the offset
  s = mid
  e = mid + left_offset + right_offset
  return im[s: e]


class SliceInput(object):
  def __init__(self, channels="1,2"):
    self.channels = torch.LongTensor([int(x) for x in channels.split(',')])

  def __call__(self, input, target=None):
    input = input[self.channels]
    return input, target


class FoldTarget(object):
  def __init__(self):
    pass

  def __call__(self, input, target=None):
    if target is not None:
      target = target.view(-1, target.size(1), target.size(1))
    return input,target


class Transpose(object):
  def __init__(self, *args):
    self.new_order = args

  def __call__(self, input, target=None):
    input = np.transpose(input, self.new_order)
    return input, target


class FoldInput(object):
  def __init__(self):
    pass

  def __call__(self, input, target=None):
    input = input.reshape(-1, input.shape[1], input.shape[1])
    return input, target


class Squeeze(object):
  def __init__(self):
    pass

  def __call__(self, input, target=None):
    input = input.squeeze(0)
    return input, target


class CropInput(object):
  def __init__(self, w, h):
    self.w, self.h = w, h

  def __call__(self, input, target=None):
    y_s = (input.shape[1] - self.h) // 2
    x_s = (input.shape[2] - self.w) // 2
    input = input[:, y_s:(y_s + self.h), x_s:(x_s + self.w)]

    return input, target


class Resize(object):
  def __init__(self, w, h):
    self.w, self.h = w, h

  def __call__(self, input, target=None):
    input = cv2.resize(input, (self.w, self.h))

    return input, target

