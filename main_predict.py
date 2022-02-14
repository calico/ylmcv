"""Main code for doing the predictions."""

import argparse
from collections import defaultdict
import importlib
import json
import math
import os
import pickle
import re
import time

import cv2

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

from dataset import BuddingDataset
from dataset import hyperstack_data_loader_wrapper

import img_aug


parser = argparse.ArgumentParser(description='Barebone testing scripts')
parser.add_argument('-test_list', type=str,
                    help='list of image files to be tested')
parser.add_argument('-model', type=str, default='model/count_model_01.py',
    help='model definition')
parser.add_argument('-checkpoint', type=str,required=True,
    help='model checkpoint')
parser.add_argument('-data_dir', type=str,
                    help='data dir')
parser.add_argument('-num_thread', type=int, default=6,
                    help='Number of threads to use for daa loading')
parser.add_argument('-num_TTA', type=int, default=1,
                    help='Number of test time augmentations')
parser.add_argument('-output_file', type=str, default=None,
                    help="save the raw result file")
parser.add_argument('-model_out_dim', type=int, default=1,
                help='Number of model output dimensions')
parser.add_argument('-num_input_channel', type=int, default=3,
                help='number of input image channels')
parser.add_argument('-z_slices', type=str, default='',
                    help='Z slices to predict')
parser.add_argument('-channels', type=str, default='',
                    help='Channels to predict')

args = parser.parse_args()

args.z_slices = [int(x) for x in args.z_slices.split(',') if x]
args.channels = [int(x) for x in args.channels.split(',') if x]

model_file = args.model.replace('/', '.')
model_file = re.sub('\.py$', '', model_file)

print('using model:', model_file)
model_module = importlib.import_module(model_file)


def predict(model, data_loader):
  """Runs the model on the given data."""

  model.eval()
  result = []

  for i, batch in enumerate(data_loader, 1):
    input_data = batch['img_stack']
    z_slices = batch['z']
    channels = batch['c']
    ids = batch['file_name']

    inputs = input_data.cuda()
    outputs = model(inputs)
    out = outputs.data.squeeze(1).cpu().numpy()

    z_slices_str = ':'.join([str(int(x)) for x in z_slices])
    channels_str = ':'.join([str(int(x)) for x in channels])

    result.append({'prediction': out, 'z':z_slices_str, 'c': channels_str,
      'id': ids[0]})

  return result


def reset_seed():
  ts = int(time.time() * 100)
  torch.manual_seed(ts)
  np.random.seed(ts % (2 ** 32 - 1))


def default_jitter_func(src_width, src_height, out_width, out_height):
  return img_aug.getImageAugMat(src_width, src_height, out_width, out_height,
                               0.88, 1.12, 0.88, 1.12, True, math.pi / 12)


def main():

  test_list = pd.read_csv(args.test_list, header=None).iloc[:,0]
  print ('number of test samples %s ' % len(test_list))

  test_meta = {'id': test_list, 'label': None}

  if hasattr(model_module, 'jitter_func'):
    jitter = model_module.jitter_func
  else:
    jitter = default_jitter_func

  warp = img_aug.WarpAffine(model_module.sample_size[1], model_module.sample_size[2],
                            jitter, 0, False, False)

  # list of image transformations
  img_tfm_list = [
      img_aug.CenterCrop(144, 144),
      warp,
      img_aug.GCN()]

  img_tfm = img_aug.Compose(img_tfm_list)

  test_set = BuddingDataset(test_meta['id'], args.data_dir,
      img_aug=img_tfm,
      z_slices=args.z_slices,channels=args.channels)

  test_data_loader = DataLoader(dataset=test_set,
    num_workers=args.num_thread, batch_size=1, shuffle=False)


  state_dict = torch.load(args.checkpoint, map_location='cuda:0')

  model_args = dict(out_dim=args.model_out_dim, dropout=0,
      num_input_channel=args.num_input_channel)
  model = model_module.Net(**model_args)

  model.load_state_dict(state_dict)
  model.cuda()

  model.batch_size = 1 # to make sure we process all the data
  all_result = []
  avg_result = []

  for i in range(args.num_TTA):
    print ("\nTesting round %d\n" % (i + 1))
    reset_seed()
    result = predict(model,
        hyperstack_data_loader_wrapper(test_data_loader))
    all_result.append(result)

  if len(all_result) > 1:
    for i in range(len(all_result[0])):
      item0 = all_result[0][i]
      prediction = [item0['prediction']]
      for j in range(1, len(all_result)):
        prediction.append(all_result[j][i]['prediction'])
      avg_prediction = np.stack(prediction, axis=0).mean(axis=0)
      avg_item = item0
      avg_item['prediction'] = avg_prediction
      avg_result.append(avg_item)
  else:
    avg_result = result

  # Save the predictions
  with open(args.output_file, "wb") as f:
    pickle.dump(avg_result, f)


if __name__ == "__main__":
  main()
