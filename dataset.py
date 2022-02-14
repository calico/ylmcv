"""Define the dataset."""

import collections
import json
import math
import os
import pickle
import re
import sys
import warnings

import cv2
import piexif

import numpy as np
import pandas as pd

import torch
import torch.utils.data as data

def _read_original_shape(jpg_path):
  exif_dict = piexif.load(jpg_path)
  try:
    md = json.loads(piexif.helper.UserComment.load(
        exif_dict["Exif"][piexif.ExifIFD.UserComment]))
    return tuple(md['stack_shape'])
  except KeyError:
    raise IOError('Image does not have stack stored in metadata')


def _read_original_order(jpg_path):
  exif_dict = piexif.load(jpg_path)
  try:
    md = json.loads(piexif.helper.UserComment.load(
        exif_dict["Exif"][piexif.ExifIFD.UserComment]))
    if 'stack_oder' in md:
        return md['stack_oder']
    return md['stack_order']
  except KeyError:
    raise IOError('Image does not have stack stored in metadata')


def _read_tile_type(jpg_path):
  exif_dict = piexif.load(jpg_path)
  md = json.loads(piexif.helper.UserComment.load(
      exif_dict["Exif"][piexif.ExifIFD.UserComment]))
  if 'tile_type' in md:
    return md['tile_type']
  else:
    return 1


def load_tiled_jpg_as_hyperstack(jpg_path):
  orig_shape = _read_original_shape(jpg_path)
  tile_type = _read_tile_type(jpg_path)
  with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      tiled_img = cv2.imread(jpg_path, cv2.IMREAD_GRAYSCALE)
  return untile(tiled_img, orig_shape)


def untile(im, shape):
  """Untile the images stored in tiled jpg."""
  # Shape is the original image shape
  h, w = shape[3:]
  size = reduce(mul, shape, 1)
  num_row = im.shape[0] // h
  num_col = im.shape[1] // w
  im = im.reshape(num_row, h, num_col, w)
  im = np.transpose(im, (0, 2, 1, 3))
  if size < im.size:
      im = im.ravel()[:size]
  im = im.reshape(*shape)
  return im


def read_pickled_images(img_pickle, limit=-1):
  imgs = []
  coded_imgs = pickle.load(open(img_pickle, "rb"))
  for i, jpg in enumerate(coded_imgs):
    img =  cv2.imdecode(jpg, 0).astype(np.float32)
    img /= 255
    imgs.append(torch.from_numpy(img).unsqueeze(0))
    if limit > 0 and i == limit -1 :
      break
  img_stack = torch.cat(imgs, dim=0)
  return img_stack


def read_jpg_images(filename, limit=-1):
  im = cv2.imread(filename)
  if im.ndim==3 and im.shape[2] == 3:
    im = np.ascontiguousarray(im[:,:,1])

  width = im.shape[1]
  height = im.shape[0]
  assert height % width == 0, "height must be mulitple of width"
  im = im.reshape((-1, width, width))
  if limit > 0:
    im = im[:limit]
  im = im.astype(np.float32)
  im /= 255
  return im


def get_z_and_channel(filename):
  m = re.search(r'c(\d+)z(\d+)_x', filename)
  if m:
    return int(m.group(1)), int(m.group(2))
  return [-1], [-1]


def read_hyperstack(filename):
  im = load_tiled_jpg_as_hyperstack(filename)
  im = im.astype(np.float32)
  im /= 255.
  return torch.from_numpy(im)


def read_img(fname, limit=-1, z_slices=[], channels=[]):

  is_hyperstack = False
  if fname.endswith(".pkl"):
    im = read_pickled_images(fname, limit)
  else:
    try:
      is_hyperstack = True
      im = load_tiled_jpg_as_hyperstack(fname)
      im = im.astype(np.float32)
      im /= 255.
      if limit > 0:
        im = im[:limit]
      if z_slices:
        im = im[:, z_slices, channels]
    except :
      im = read_jpg_images(fname, limit)
      if im.ndim == 3:
        t, h, w = im.shape
        im = im.reshape(t, 1, 1, h, w)
      else:
        t, h, w, c = im.shape
        im = np.transpose(im, (0, 3, 1, 2))
        im = im.reshape(t, -1, 1, h, w)
    im = torch.from_numpy(im)

  return im, is_hyperstack


class BuddingDataset(data.Dataset):
  """ define the Dataset class"""
  def __init__(self, ids, data_dir, input_transform=None,
      z_slices=[], channels=[], img_aug=None):
    super(BuddingDataset, self).__init__()
    self.ids = ids
    self.input_transform = input_transform
    self.data_dir = data_dir
    self.z_slices = z_slices
    self.channels = channels
    self.img_aug = img_aug

  def __getitem__(self, index):
    fname = self.ids[index]
    limit = -1

    data, is_hyperstack = read_img(
        os.path.join(self.data_dir, fname),
        limit,
        z_slices=self.z_slices,
        channels=self.channels)

    z_slices = self.z_slices if self.z_slices else np.arange(data.shape[1],
        dtype=np.int32)
    channels = self.channels if self.channels else np.arange(data.shape[2],
        dtype=np.int32)
    if not is_hyperstack:
      z_slices, channels = get_z_and_channel(fname)

    if self.input_transform:
      if is_hyperstack:
        shape = data.shape
        data = data.reshape(-1, *shape[-2:])
        data, _ = self.input_transform(data, None)
        new_shape = data.shape
        data = data.reshape(*shape[:-2], *new_shape[1:])
      else:
        data, _ = self.input_transform(data, None)

    if self.img_aug:
      data = data.numpy()
      shape = data.shape
      data = data.reshape(-1, *shape[-2:])
      data = np.transpose(data, (1, 2, 0))
      data, _ = self.img_aug(data, None)
      data = np.transpose(data, (2, 0, 1))
      new_shape = data.shape
      data = data.reshape(*shape[:-2], *new_shape[1:])
      data = torch.from_numpy(data)

    data_item = {'img_stack': data, 'file_name': fname, 'z_slices': z_slices,
        'channels': channels}
    return data_item

  def __len__(self):
    return len(self.ids)


def from_np_array(s):
  s = re.sub('[\[\]\n]', '', s)
  s = re.sub(' +', ' ', s)
  s = re.sub('^ +', '', s)
  val = np.array([int(x) for x in s.split(' ')])
  return val


def stack_image(im):
  """input in ntchw format."""
  mid = im
  left = np.concatenate([im[:, 0:1], im[:, :-1]], axis=1)
  right = np.concatenate([im[:, 1:], im[:, -1:]], axis=1)
  stack_im = np.concatenate([left, mid, right], axis=2)
  return torch.from_numpy(stack_im)


def hyperstack_data_loader_wrapper(dataloader, stack=False):
  """A generator wrapper for the data loader to iterate over the z slices."""
  for data_item in dataloader:
    # im should be in ntzchw format
    img_stack = data_item['img_stack']
    for b in range(img_stack.shape[0]):
      im = img_stack[b]
      z_slices = data_item['z_slices'][b]
      channels = data_item['channels'][b]
      for i, z in enumerate(z_slices):
        for j, c in enumerate(channels):
          # returns ntzhw (n always be one)
          im_slice = im[None, :, i: (i+1), j, ]
          out = {'img_stack': im_slice, 'z': [z], 'c': [c],
              'file_name': data_item['file_name']}
          if stack:
            stack_im_slice = stack_image(im_slice)
            out['stacked_image'] = stack_im_slice
          for k, v in data_item.items():
            if k not in out:
              out[k] = v
          yield out
