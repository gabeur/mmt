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
"""Config parser.

Code based on the implementation of "Collaborative Experts":
https://github.com/albanie/collaborative-experts
"""

import functools
import logging
import operator
import os
import pathlib
import pprint

import torch
from utils import get_last_checkpoint_path
from utils import read_json
from utils import write_json

logger = logging.getLogger(__name__)


class ConfigParser:
  """Config parser."""

  def __init__(self, args, options=''):

    if args.resume:
      msg_cfg = 'If resuming experiment then no config should be provided'
      assert args.config is None, msg_cfg
      msg_cfg = 'If resuming experiment then no checkpoint should be provided'
      assert args.load_checkpoint is None, msg_cfg
      exp_dir = pathlib.Path(args.resume)
      checkpoint_path = get_last_checkpoint_path(exp_dir)
      self.resume = checkpoint_path
      self.cfg_fname = exp_dir / 'config.json'

    else:
      msg_no_cfg = 'Config file must be specified'
      assert args.config is not None, msg_no_cfg
      self.resume = None
      self.cfg_fname = pathlib.Path(args.config)

      if args.load_checkpoint:
        checkpoint_path = args.load_checkpoint
        self.resume = checkpoint_path

    if args.only_eval:
      self.only_eval = True
    else:
      self.only_eval = False

    # load config file and apply custom cli options
    config = read_json(self.cfg_fname)
    self._config = _update_config(config, options, args)

    if 'exp_name' in self.config.keys():
      exper_name = self.config['exp_name']
    else:
      exper_name = pathlib.Path(args.config).stem
      self._config['exp_name'] = exper_name

    # set save_dir where trained model and log will be saved.
    if 'save_dir' in self.config['trainer'].keys():
      save_dir = pathlib.Path(self.config['trainer']['save_dir'])
    else:
      save_dir = pathlib.Path.cwd() / 'exps' / exper_name
      self._config['trainer']['save_dir'] = str(save_dir)

    self._save_dir = save_dir
    self._log_dir = save_dir
    self._web_dirs = [save_dir / 'visualisations']
    self._exper_name = exper_name
    self._args = args

    if 'external_save_dir' in self.config['trainer'].keys():
      external_save_dir = pathlib.Path(
          self.config['trainer']['external_save_dir'])
      self._web_dirs.append(external_save_dir / 'visualisations')
    else:
      external_save_root = pathlib.Path.cwd() / 'external_save_dir'
      if external_save_root.exists():
        external_save_dir = external_save_root / 'exps' / exper_name
        self._config['trainer']['external_save_dir'] = str(save_dir)
        self._web_dirs.append(external_save_dir / 'visualisations')

    self.save_dir.mkdir(parents=True, exist_ok=True)
    self.log_dir.mkdir(parents=True, exist_ok=True)

    logpath = save_dir / 'log.txt'
    if args.verbose:
      logging.basicConfig(
          level=os.environ.get('LOGLEVEL', 'DEBUG'),
          # format='%(relativeCreated)6d %(threadName)s %(message)s')
          # format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
          format='%(message)s')
    else:
      logging.basicConfig(
          level=os.environ.get('LOGLEVEL', 'INFO'),
          # format='%(relativeCreated)6d %(threadName)s %(message)s')
          # format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
          handlers=[logging.FileHandler(logpath),
                    logging.StreamHandler()],
          format='%(message)s')

    logger.info('Experiment directory: %s', save_dir)

    if args.device == 'cpu':
      os.environ['CUDA_VISIBLE_DEVICES'] = ''
    elif args.device:
      os.environ['CUDA_VISIBLE_DEVICES'] = args.device
      logger.debug('CUDA_VISIBLE_DEVICES: %s',
                   os.environ['CUDA_VISIBLE_DEVICES'])

    n_gpu = torch.cuda.device_count()
    logger.debug('n_gpu = torch.cuda.device_count(): %d (nb of gpus available)',
                 n_gpu)

    # save updated config file to the checkpoint dir
    write_json(self.config, self.save_dir / 'config.json')

    # Print the configuration
    logging.debug(pprint.pformat(self.config))

  def init(self, name, module, *args, **kwargs):
    """Finds a function handle with the name given as 'type' in config."""
    module_name = self[name]['type']
    module_args = dict(self[name]['args'])
    msg = 'Overwriting kwargs given in config file is not allowed'
    assert all([k not in module_args for k in kwargs]), msg
    module_args.update(kwargs)
    return getattr(module, module_name)(*args, **module_args)

  def __getitem__(self, name):
    return self.config[name]

  def get(self, name, default):
    return self.config.get(name, default)

  # setting read-only attributes
  @property
  def config(self):
    return self._config

  @property
  def save_dir(self):
    return self._save_dir

  @property
  def log_dir(self):
    return self._log_dir

  @property
  def exper_name(self):
    return self._exper_name

  @property
  def web_dirs(self):
    return self._web_dirs

  def __repr__(self):
    return pprint.PrettyPrinter().pprint.pformat(self.__dict__)


# helper functions used to update config dict with custom cli options
def _update_config(config, options, args):
  for opt in options:
    value = getattr(args, _get_opt_name(opt.flags))
    if value is not None:
      _set_by_path(config, opt.target, value)
  return config


def _get_opt_name(flags):
  for flg in flags:
    if flg.startswith('--'):
      return flg.replace('--', '')
  return flags[0].replace('--', '')


def _set_by_path(tree, keys, value):
  """Set a value in a nested object in tree by sequence of keys."""
  _get_by_path(tree, keys[:-1])[keys[-1]] = value


def _get_by_path(tree, keys):
  """Access a nested object in tree by sequence of keys."""
  return functools.reduce(operator.getitem, keys, tree)
