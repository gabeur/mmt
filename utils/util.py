# Copyright 2020 Valentin Gabeur
# Copyright 2020 Samuel Albanie, Yang Liu and Arsha Nagrani
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities.

Code based on the implementation of "Collaborative Experts":
https://github.com/albanie/collaborative-experts
"""

import collections
import json
import logging
import os
import pathlib
import pickle
import re
import time

import numpy as np
import torch
from typeguard import typechecked

logger = logging.getLogger(__name__)


@typechecked
def compress_predictions(query_masks: np.ndarray,
                         sims: np.ndarray,
                         topk: int = 10):
  """Flatten the predictions and keep the top k."""

  # We store the indices of the top-k predictions, rather than the full
  # similarity matrix, to reduce storage requirements.
  # NOTE: The similarity matrix contains `num_queries x num_videos` elements,
  # where
  # `num_queries = num_videos x max_num_queries_per_video`.  We first mask out
  # locations in the similarity matrix that correspond to invalid queries (these
  # are
  # produced by videos with fewer than `max_num_queries_per_video`
  # descriptions).

  # validate the input shapes
  assert query_masks.ndim == 2, 'Expected query_masks to be a matrix'
  query_num_videos, query_max_per_video = query_masks.shape
  sims_queries, sims_num_videos = sims.shape
  msg = (
      f'Expected sims and query masks to represent the same number of videos '
      f'(found {sims_num_videos} v {query_num_videos}')
  assert query_num_videos == sims_num_videos, msg
  msg = (
      f'Expected sims and query masks to represent the same number of queries '
      f'(found {sims_queries} v {query_num_videos * query_max_per_video}')
  assert query_max_per_video * query_num_videos == sims_queries, msg

  valid_sims = sims[query_masks.flatten().astype(np.bool)]
  ranks = np.argsort(-valid_sims, axis=1)
  return ranks[:, :topk]


def get_last_checkpoint_path(exp_dir):
  """Get the path of the last saved checkpoint."""
  last_checkpoint_path = None
  highest_epoch = -1
  for filename in os.listdir(exp_dir):
    is_ckpt = re.search(r'checkpoint-epoch([0-9]+)\.pth$', filename)
    if is_ckpt:
      ckpt_epoch = int(is_ckpt.group(1))
      if ckpt_epoch > highest_epoch:
        highest_epoch = ckpt_epoch
        last_checkpoint_path = os.path.join(exp_dir, filename)
  return last_checkpoint_path


def verbose(epoch, metrics, mode, name='TEST'):
  """Print the metrics."""
  r1, r5, r10, r50 = metrics['R1'], metrics['R5'], metrics['R10'], metrics[
      'R50']
  medr, meanr = metrics['MedR'], metrics['MeanR']
  msg = f'[{mode}]{name:s} epoch {epoch}, R@1: {r1:.1f}'
  msg += f', R@5: {r5:.1f}, R@10 {r10:.1f}, R@50 {r50:.1f}'
  msg += f' MedR: {medr:g}, MeanR: {meanr:.1f}'
  print(msg)


def memcache(path):
  """Read data from file."""
  suffix = pathlib.Path(path).suffix
  if suffix in {'.pkl', '.pickle'}:
    res = pickle_loader(path)
  elif suffix == '.npy':
    res = np_loader(path)
  else:
    raise ValueError(f'unknown suffix: {suffix}')
  return res


def read_json(fname):
  """Read from a json file."""
  with fname.open('rt') as handle:
    return json.load(handle, object_hook=collections.OrderedDict)


def write_json(content, fname):
  """Write to a json file."""
  with fname.open('wt') as handle:
    json.dump(content, handle, indent=4, sort_keys=False)


def pickle_loader(pkl_path):
  """Read from a pkl file."""
  tic = time.time()
  logger.debug('loading features from %s', pkl_path)
  with open(pkl_path, 'rb') as f:
    data = pickle.load(f, encoding='latin1')
  logger.debug('done in {:.3f}s'.format(time.time() - tic))
  return data


def np_loader(np_path, l2norm=False):
  """Read from an np file."""
  tic = time.time()
  logger.debug('loading features from %s', np_path)
  with open(np_path, 'rb') as f:
    data = np.load(f, encoding='latin1', allow_pickle=True)
  logger.debug('done in {:.3f}s'.format(time.time() - tic))
  if isinstance(data, np.ndarray) and data.size == 1:
    data = data[()]  # handle numpy dict storage convnetion
  if l2norm:
    logger.debug('L2 normalizing features')
    if isinstance(data, dict):
      for key in data:
        feats_ = data[key]
        feats_ = feats_ / max(np.linalg.norm(feats_), 1E-6)
        data[key] = feats_
    elif data.ndim == 2:
      data_norm = np.linalg.norm(data, axis=1)
      data = data / np.maximum(data_norm.reshape(-1, 1), 1E-6)
    else:
      raise ValueError('unexpected data format {}'.format(type(data)))
  return data


def compute_dims(config):
  """Get the dimensions of the different experts features."""
  ordered = sorted(config['experts']['modalities'])

  dims = []
  for expert in ordered:
    if expert == 's3d':
      in_dim = 1024
      idx = 1
    elif expert == 'vggish':
      in_dim = 128
      idx = 2
    elif expert == 'face':
      in_dim = config['experts']['face_dim']
      idx = 3
    elif expert == 'audio':
      in_dim = 128
      idx = 4
    elif expert == 'rgb':
      in_dim = 2048
      idx = 5
    elif expert == 'speech':
      in_dim = 300
      idx = 6
    elif expert == 'ocr':
      in_dim = 300
      idx = 7
    elif expert == 'flow':
      in_dim = 1024
      idx = 8
    elif expert == 'scene':
      in_dim = 2208
      idx = 9

    elif expert == 'audio_c':
      in_dim = 128
      idx = 1
    elif expert == 'face_c':
      in_dim = config['experts']['face_dim']
      idx = 2
    elif expert == 'i3d':
      in_dim = 1024
      idx = 3
    elif expert == 'resnext101_32x48d':
      in_dim = 2048
      idx = 4
    elif expert == 'senet154':
      in_dim = 2048
      idx = 5
    elif expert == 'ocr_c':
      in_dim = 300
      idx = 6
    elif expert == 'r2p1d':
      in_dim = 512
      idx = 7
    elif expert == 's3dg':
      in_dim = 1024
      idx = 8
    elif expert == 'densenet161':
      in_dim = 2208
      idx = 9
    elif expert == 'speech_c':
      in_dim = 300
      idx = 10
    elif expert == 'r2p1dk':
      in_dim = 512
      idx = 11

    elif expert == 'i3d_logits':
      in_dim = 400
      idx = 12
    elif expert == 'resnext101_32x48d_logits':
      in_dim = 1000
      idx = 13
    elif expert == 'senet154_logits':
      in_dim = 1000
      idx = 14
    elif expert == 'r2p1d_logits':
      in_dim = 359
      idx = 15
    elif expert == 's3dg_logits':
      in_dim = 512
      idx = 16
    elif expert == 'densenet161_logits':
      in_dim = 365
      idx = 17
    elif expert == 'r2p1dk_logits':
      in_dim = 400
      idx = 18

    dims.append((expert, {'dim': in_dim, 'idx': idx}))
  expert_dims = collections.OrderedDict(dims)

  return expert_dims


def mkdirs(paths):
  """create empty directories if they don't exist."""
  if isinstance(paths, list) and not isinstance(paths, str):
    for path in paths:
      mkdir(path)
  else:
    mkdir(paths)


def mkdir(path):
  """create a single empty directory if it didn't exist."""
  if not os.path.exists(path):
    os.makedirs(path)


def get_len_sequences(x):
  """Return the length of the zero padded lines."""

  axis = 1  # Check for first occurence along second dim

  # Add a zero at the end of each line
  b, l = x.size()
  y = torch.zeros(b, l + 1)
  y[:, :l] = x

  # Truth table where there are zeros
  zs = y == 0

  # Truth table of the first zero occurence on a line
  fzs = (zs.cumsum(axis) == 1)

  _, indices = fzs.max(axis)

  return indices


def get_list_of_files(dir_name):
  """Return the list of files contained in a directory."""
  listoffiles = list()
  for (dirpath, _, filenames) in os.walk(dir_name):
    listoffiles += [os.path.join(dirpath, file) for file in filenames]
  return sorted(listoffiles)


def default_to_regular(d):
  if isinstance(d, collections.defaultdict):
    d = {k: default_to_regular(v) for k, v in d.items()}
  return d


def get_expert_paths(data_dir):
  """Get the filepaths containing the expert features."""
  nested_dict = lambda: collections.defaultdict(nested_dict)
  expert_paths = nested_dict()
  path_list = get_list_of_files(data_dir)
  for path in path_list:
    relpath = os.path.relpath(path, data_dir)
    dir_name = path.split('/')[-2]
    if dir_name.startswith('aggregated'):
      basename = os.path.basename(path)
      mod_name = basename.split('-')[0].lower()

      if mod_name in ['ocr', 'scene', 'face', 'audio', 'speech']:
        mod_name = mod_name + '_c'

      if 'r2p1d-ig65m-kinetics' in basename:
        mod_name = 'r2p1dk'

      if '-logits' in basename:
        mod_name += '_logits'

      if basename in [
          'Audio_MSRVTT_new.pickle', 'vggish-audio-raw.pickle',
          'vggish-raw.pickle'
      ]:
        mod_name = 'audio_c'
        expert_paths[mod_name]['fixed_seg'] = relpath
        continue

      if basename in ['facefeats-avg.pickle', 'face-avg.pickle']:
        mod_name = 'face_c'
        expert_paths[mod_name]['fixed_seg'] = relpath
        continue

      if basename in [
          'ocr-raw.pickle', 'ocr-w2v.pkl', 'ocr-feats.pkl', 'ocr-w2v.pickle'
      ]:
        mod_name = 'ocr_c'
        expert_paths[mod_name]['fixed_seg'] = relpath
        continue

      if basename in [
          'speech-w2v.pickle', 'goog_w2v-speech-raw.pickle', 'stt_w2v.pickle'
      ]:
        mod_name = 'speech_c'
        expert_paths[mod_name]['fixed_seg'] = relpath
        continue

      if basename.endswith('-max.pickle') or basename.endswith(
          '-max-logits.pickle'):
        expert_paths[mod_name]['max'] = relpath
        continue
      if basename.endswith('-avg.pickle') or basename.endswith(
          '-avg-logits.pickle'):
        expert_paths[mod_name]['avg'] = relpath
        continue
      if basename.endswith('-fixed_seg.pickle') or basename.endswith(
          '-fixed_seg-logits.pickle'):
        expert_paths[mod_name]['fixed_seg'] = relpath
        continue

    elif os.path.basename(path).startswith('raw-captions.'):
      expert_paths['raw_captions'] = relpath
    elif os.path.basename(path).startswith('raw-captions-test2.'):
      expert_paths['raw_captions_test2'] = relpath

  expert_paths = default_to_regular(expert_paths)

  return expert_paths


def get_hparams_from_config(config):
  """Create a dict of all the hyperparameters from the configuration."""

  if isinstance(config, str):
    assert os.path.exists(config), f'The path {config} do not exist!'
    config = read_json(pathlib.Path(config))

  hparams = {}
  hparams['seed'] = config['seed']

  if 'mix' not in config['train_sets'][0]['args']:
    return hparams

  mix = config['train_sets'][0]['args']['mix']
  pretraining = len(config['train_sets']
                   ) > 1 and config['train_sets'][0]['args']['until_epoch'] > 0
  if pretraining:
    hparams['ptrn_epochs'] = config['train_sets'][0]['args']['until_epoch']
    for data_dic in mix:
      hparams[f'weight_{data_dic["dataset_name"]}'] = data_dic['mix_weight']
  else:
    ftn_mix = config['train_sets'][-1]['args']['mix']
    for data_dic in ftn_mix:
      hparams[f'weight_{data_dic["dataset_name"]}'] = 1
    hparams['ptrn_epochs'] = 0

  if 'query_suffling' in config['train_sets'][0]['args']['mix'][0]:
    hparams['query_shuffling'] = config['train_sets'][0]['args']['mix'][0][
        'query_shuffling']

  for mod in config['experts']['modalities']:
    hparams[f'mod_{mod}'] = 1

  hparams['nb_mods'] = len(config['experts']['modalities'])

  use_bert = config['arch']['args']['vid_cont'] == 'bert'
  if use_bert:
    hparams['vid/num_hidden_layers'] = config['arch']['args'][
        'vid_bert_params']['num_hidden_layers']
    hparams['vid/num_attention_heads'] = config['arch']['args'][
        'vid_bert_params']['num_attention_heads']
    hparams['vid/hidden_dropout'] = config['arch']['args']['vid_bert_params'][
        'hidden_dropout_prob']
    hparams['vid/attention_dropout'] = config['arch']['args'][
        'vid_bert_params']['attention_probs_dropout_prob']
    hparams['vid/max_position_embeddings'] = config['arch']['args'][
        'vid_bert_params']['max_position_embeddings']
    hparams['vid/pos_enc'] = config['arch']['args']['pos_enc']
    hparams['vid/out_tok'] = config['arch']['args']['out_tok']

  use_txt_bert = config['arch']['args']['txt_agg'].startswith('bert')
  if use_txt_bert and 'txt_bert_params' in config['arch']['args']:
    hparams['txt/hidden_dropout'] = config['arch']['args']['txt_bert_params'][
        'hidden_dropout_prob']
    hparams['txt/attention_dropout'] = config['arch']['args'][
        'txt_bert_params']['attention_probs_dropout_prob']

  hparams['keep_missing_modalities'] = config['arch']['args'][
      'keep_missing_modalities']

  hparams['remove_stop_words'] = False
  if 'remove_stop_words' in mix[0]:
    if mix[0]['remove_stop_words']:
      hparams['remove_stop_words'] = True

  for data_dic in config['train_sets'] + config[
      'continuous_eval_sets'] + config['final_eval_sets']:
    if 'n_pairs' in data_dic['args'].keys() and data_dic['args']['n_pairs'] > 1:
      hparams['n_pairs'] = data_dic['args']['n_pairs']

  hparams['nb_modalities'] = len(config['experts']['modalities'])

  hparams['txt_inp'] = config['arch']['args']['txt_inp']
  hparams['txt_agg'] = config['arch']['args']['txt_agg']
  hparams['txt_pro'] = config['arch']['args']['txt_pro']
  hparams['txt_wgh'] = config['arch']['args']['txt_wgh']
  hparams['vid_wgh'] = config['arch']['args']['vid_wgh']
  hparams['vid_cont'] = config['arch']['args']['vid_cont']
  hparams['vid_inp'] = config['arch']['args']['vid_inp']
  hparams['lr'] = config['optimizer']['args']['lr']
  hparams['weight_decay'] = config['optimizer']['args']['weight_decay']
  if 'gamma' in config['lr_scheduler']['args']:
    hparams['gamma'] = config['lr_scheduler']['args']['gamma']
  hparams['epochs'] = config['trainer']['epochs']
  hparams['loss'] = config['loss']['type']
  if 'margin' in config['loss']['args']:
    hparams['margin'] = config['loss']['args']['margin']
  hparams['batch_size'] = config['train_sets'][0]['args']['batch_size']
  hparams['max_samples_per_epoch'] = config['trainer']['max_samples_per_epoch']
  hparams['max_text_words'] = config['train_sets'][0]['args']['mix'][0][
      'max_text_words']
  hparams['n_gpu'] = config['n_gpu']

  return hparams
