"""Utility functions for image augmentations."""

import math
import numpy as np
import cv2


class Compose(object):
  """Takeing a list of transforms and combine them together."""
  def __init__(self, transforms):
    self.transforms = transforms

  def __call__(self, data, target):
    for t in self.transforms:
        data, target = t(data, target)
    return data, target


class GCN(object):
  """Global contrast normalization."""

  def __init__(self, eps=1e-5, channel_last=True):
    self.eps = eps
    self.channel_last = channel_last

  def __call__(self, data, target):
    image = data
    if self.channel_last:
      img_mean = np.mean(image, axis=(0, 1)).reshape(1, 1, -1)
      img_std = np.std(image, axis=(0, 1)).reshape(1, 1, -1)
    else:
      img_mean = np.mean(image, axis=(1,2)).reshape(-1, 1, 1)
      img_std = np.std(image, axis=(1,2)).reshape(-1, 1, 1)
    data = (image - img_mean)/(img_std + self.eps)
    return data, target


class CenterCrop(object):
  """Crop the center of the image."""

  def __init__(self, width, height, channel_last=True):
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
    hs = max(0, hs)
    ws = max(0, ws)
    if self.channel_last:
      im = im[hs: he, ws: we]
    else:
      im = im[:, hs: he, ws: we]
    if isinstance (im, np.ndarray):
      return np.ascontiguousarray(im), target
    else:
      return im.contiguous(), target


class Normalize(object):
  """Normalize the image."""

  def __init__(self, mean=[0.5], std=[1.], max_pixel_value=255.0):
    self.mean = mean
    self.std = std
    self.max_pixel_value = max_pixel_value

  def __call__(self, im, target):
    normalized_img = normalize_stack(im, self.mean,
        self.std, self.max_pixel_value)
    return normalized_img, target


def getTransform(
        srcWidth, srcHeight, outWidth=None, outHeight=None,
        xStretch=1, xShear=0,
        yStretch=1, yShear=0,
        rotate=0,
        hFlip=False, vFlip=False,
        xCrop=None, yCrop=None
):

  c00, c01, c10, c11 = 1, 0, 0, 1
  c00 = c00 * xStretch
  c11 = c11 * yStretch

  if hFlip:
    c00 = -c00

  if vFlip:
    c11 = -c11

  c01 = xShear * c00 + c01
  c11 = xShear * c10 + c11

  c10 = yShear * c00 + c10
  c11 = yShear * c01 + c11

  trMat = np.array([[c00, c01], [c10, c11]])

  c = math.cos(rotate)
  s = math.sin(rotate)
  rotMat = np.array([[c, s], [-s, c]])
  trMat = np.dot(trMat, rotMat)

  # compute the coordinates of frames after transformation
  srcX = srcWidth - 1
  srcY = srcHeight - 1
  src = np.array([[0, 0], [srcX, 0], [0, srcY], [srcX, srcY]])
  dst = np.dot(src, trMat)
  # minimum coordinates, maximum coordinates
  minDst = dst.min(axis=0)
  maxDst = dst.max(axis=0)

  dstSz = maxDst - minDst + 1
  dstWidth = dstSz[0]
  dstHeight = dstSz[1]

  if outWidth is None:
    outWidth = srcWidth

  if outHeight is None:
    outHeight = srcHeight

  if xCrop is None:
    xCrop = outWidth / 2
  if yCrop is None:
    yCrop = outHeight / 2

  dstRoiX = dstWidth / 2 - xCrop
  dstRoiY = dstHeight / 2 - yCrop

  return np.array([[trMat[0, 0], trMat[1, 0], -minDst[0] - dstRoiX],
                       [trMat[0, 1], trMat[1, 1], -minDst[1] - dstRoiY]])


def getImageAugMat(srcWidth, srcHeight, outWidth, outHeight,
                    crpStartX=0.7, crpEndX=0.9, crpStartY=0.7, crpEndY=0.9,
                    centerOnly=False, rotationRange=math.pi,
                    keepAspectRatio=False, hFlip=True, vFlip=False):
  """Combine the desired augmentation into a single affine matrix."""

  szRatioX = outWidth / srcWidth
  szRatioY = outHeight / srcHeight

  # random number array
  d = np.random.rand(7)

  if hFlip:
    hFlip = d[0] < 0.5
  if vFlip:
    vFlip = d[1] < 0.5

  crpSzX = crpStartX + d[2] * (crpEndX - crpStartX)
  crpSzY = crpStartY + d[3] * (crpEndY - crpStartY)
  if keepAspectRatio:
    crpSzY = crpSzX

  xOffset = (1.0 - crpSzX) / 2
  yOffset = (1.0 - crpSzY) / 2

  if not centerOnly:
    xOffset = (1.0 - crpSzX) * d[4]
    yOffset = (1.0 - crpSzY) * d[5]

  xStretch = szRatioX / crpSzX
  yStretch = szRatioY / crpSzY
  xCrop = (0.5 - xOffset) * srcWidth * xStretch
  yCrop = (0.5 - yOffset) * srcHeight * yStretch
  rotation = (d[6] - 0.5) * 2 * rotationRange
  config = {
      'rotate': rotation,
      'hFlip': hFlip,
      'vFlip': vFlip,
      'xStretch': xStretch,
      'yStretch': yStretch,
      'xCrop': xCrop,
      'yCrop': yCrop
  }
  return getTransform(srcWidth, srcHeight, outWidth, outHeight, **config)


def translate(xOffset, yOffset):
  return np.array([[1, 0, xOffset], [0, 1, yOffset]])

# merge a table of transforms

def rotate(theta, x=0, y=0):
  tr1 = np.array([[1, 0, -x], [0, 1, -y]])
  c = math.cos(theta)
  s = math.sin(theta)
  rotMat = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
  tr2 = np.array([[1, 0, x], [0, 1, y], [0, 0, 1]])
  return np.dot(tr1, np.dot(rotmat, tr2))[0:2]


def hFlip():
  return np.array([[-1, 0, 0], [0, 1, 0]])


def vFlip():
  return np.array([[1, 0, 0], [0, -1, 0]])


def scale(xscale, yscale):
  return np.array([[xscale, 0, 0], [0, yscale, 0]])


def gaussianNoise(im, amount):
  image2 = np.random.randn(im.shape)
  im += image2 * (amount * im.std() / image2.std())
  return im



def warpAffineCV(im, rot_mat, out_shape=None, border_mode='replicte', fille_value=0):
  """Using opencv warpAffine to perform the rotation and distortion."""
  if out_shape is None:
    out_shape = im.shape[1], im.shape[0]
  w, h = out_shape

  num_channel = 1
  if im.ndim == 3:
    num_channel = im.shape[2]

  if num_channel > 512: # opencv cannonly do 512 channels
    out_list = []
    num_chunks = math.ceil(num_channel / 512)
    for i in range(num_chunks):
      s = i * 512
      e = s + 512
      out = cv2.warpAffine(im[:, :, s:e], rot_mat, out_shape,
          borderMode=cv2.BORDER_REPLICATE)
      if out.ndim == 2:
        out = out[:, :, None]
      out_list.append(out)
    out_img = np.concatenate(out_list, axis=2)
  else:
    out_img = cv2.warpAffine(im, rot_mat, out_shape,
          borderMode=cv2.BORDER_REPLICATE)
  return out_img.astype(np.float32)


class WarpAffine(object):
  """WarpAffine using numpy as input and channel last, NHWC format."""

  def __init__(self, outWidth, outHeight,
               jitterFunc, default=0,
               processTarget=False,
               targetIsCoord=False,
               targetWidth=None):

    self.outWidth = outWidth
    self.outHeight = outHeight
    self.jitterFunc = jitterFunc
    self.processTarget = processTarget
    self.targetIsCoord = targetIsCoord
    self.default = default
    if targetWidth is None:
      self.targetWidth = outWidth
      self.targetHeight = outHeight
      self.targetScale = 1
      self.doTargetScale = False
    else:
      self.targetWidth = targetWidth
      self.targetScale = targetWidth * 1.0 / outWidth
      self.targetHeight = round(self.targetScale * outHeight)
      self.doTargetScale = True
      self.targetScaleMat = np.array([[self.targetScale, 0, 0],
        [0, self.targetScale, 0],
        [0, 0, 1]])
      self.reverseTargetScaleMat = np.array([[1. / self.targetScale, 0, 0],
        [0, 1./ self.targetScale, 0],
        [0, 0, 1]])


  def __call__(self, im, target):
    assert im.ndim == 2 or im.ndim == 3, "image dimension must be 2 or 3"

    tgt = target

    warpMat = self.jitterFunc(im.shape[1], im.shape[0],
                              self.outWidth, self.outHeight)

    dst = warpAffineCV(im, warpMat, (self.targetWidth, self.targetHeight))

    if (self.processTarget):
      if self.targetIsCoord:
        tgt = np.dot(target, warpMat[:,:2].T) + warpMat[:, 2]
      else:
        if self.doTargetScale:
          tmpWarpMat = np.eye(3)
          tmpWarpMat[:2] = warpMat
          warpMat = self.targetScaleMat.dot(
              tmpWarpMat).dot(self.reverseTargetScaleMat)
          warpMat = warpMat[:2]
          tgt = warpAffineCV(target, warpMat,
              (self.targetWidth, self.targetHeight),)
    return dst, tgt
