import cv2
import numpy as np
import torch

class Compose(object):
  def __init__(self, transforms):
    self.transforms = transforms

  def __call__(self, data, target):
    for t in self.transforms:
        data, target = t(data, target)
    return data, target


class CenterCrop(object):
  def __init__(self, width, height, channel_last=False):
    self.width = width
    self.height = height
    self.channel_last = channel_last

  def __call__(self, im, target):
    if self.channel_last:
      h, w = im.shape[:2]
    else:
      h, w = im.shape[-2:]
    hs = (h - self.height) // 2
    he = hs + self.height
    ws = (w - self.width) // 2
    we = ws + self.width
    if self.channel_last:
      im = im[hs: he, ws: we]
    else:
      im = im[:, hs: he, ws: we]
    if isinstance (im, np.ndarray):
      return np.ascontiguousarray(im), target
    else:
      return im.contiguous(), target


class Normalize(object):
  def __init__(self, mean=[0.5], std=[1.], max_pixel_value=255.0):
    self.mean = mean
    self.std = std
    self.max_pixel_value = max_pixel_value

  def __call__(self, im, target):
    normalized_img = normalize_stack(im, self.mean,
        self.std, self.max_pixel_value)
    return normalized_img, target


class Resize(object):
  def __init__(self, size, resize_target=False):
    self.size = size
    self.resize_target = resize_target

  def __call__(self, im, target):
    im = resize_stack(im, self.size)
    if self.resize_target:
      target = resize_stack(target, self.size)
    return im, target


class TimeStack(object):
  def __init__(self, expand_dim=True):
    self.expand_dim = expand_dim

  def __call__(self, im, target):
    im = stack_image(im, self.expand_dim)
    return im, target


def stack_image(im, expand_dim=True):
  """input in tzchw format."""
  im = np.expand_dims(im, -3) if expand_dim else im
  left = np.concatenate([im[0:1], im[:-1]], axis=0)
  right = np.concatenate([im[1:], im[-1:]], axis=0)
  stack_im = np.concatenate([left, im, right], axis=-3)
  return stack_im


def resize_stack(im, size):
  """Input is in hwc format."""
  h, w, c = im.shape
  if c <= 512:
    return cv2.resize(im, size)
  chunks = []
  s = 0
  while (s < c):
    chunks.append(cv2.resize(im[:, :, s: (s + 512)], size))
    s += 512
  return np.concatenate(chunks, axis=2)


def normalize_stack(img, mean, std, max_pixel_value=255.0):
    mean = np.array(mean, dtype=np.float32)
    mean *= max_pixel_value

    std = np.array(std, dtype=np.float32)
    std *= max_pixel_value

    denominator = np.reciprocal(std, dtype=np.float32)

    img = img.astype(np.float32)
    img -= mean
    img *= denominator
    return img
