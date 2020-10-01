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
"""Logic for training process.

Code based on the implementation of "Collaborative Experts":
https://github.com/albanie/collaborative-experts
"""
import abc
import collections
import json
import logging
import os
import re
import time

from numpy import inf
from tensorboardX import SummaryWriter
import torch
from utils.perf_log_utils import update_perf_log
from utils.timing_utils import AverageMeter
from utils.util import get_hparams_from_config
from utils.util import get_last_checkpoint_path

logger = logging.getLogger(__name__)


class BaseTrainer:
  """Base class for all trainers."""

  def __init__(self, model, loss, metrics, optimizer, lr_scheduler, config):
    self.config = config
    self.hparams = get_hparams_from_config(self.config)

    # setup GPU device if available, move model into configured device
    self.device, device_ids = self._prepare_device(config['n_gpu'])
    self.model = model.to(self.device)
    if len(device_ids) > 1:
      self.model = torch.nn.DataParallel(model, device_ids=device_ids)

    self.loss = loss
    self.metrics = metrics
    self.optimizer = optimizer
    self.lr_scheduler = lr_scheduler

    self.exp_dir = config.save_dir
    self.checkpoint_dir = config.save_dir
    self.perf_log_path = os.path.join(config.save_dir, 'perf_log.txt')
    self.info_checkpoint_path = os.path.join(config.save_dir,
                                             'info_checkpoint.txt')
    self.monitoring_path = os.path.join(config.save_dir, 'monitoring.json')

    cfg_trainer = config['trainer']
    self.epochs = cfg_trainer['epochs']
    self.save_period = cfg_trainer['save_period']
    self.monitor = cfg_trainer.get('monitor', 'off')

    self.timer = AverageMeter()

    # configuration to monitor model performance and save best
    if self.monitor == 'off':
      self.mnt_mode = 'off'
      self.mnt_best = 0
    elif self.monitor.startswith('given_epoch'):
      self.mnt_mode, self.given_epoch = self.monitor.split()
      assert self.mnt_mode in ['given_epoch']
      self.mnt_best = 0
      self.given_epoch = int(self.given_epoch)
    else:
      self.mnt_mode, self.mnt_metric = self.monitor.split()
      assert self.mnt_mode in ['min', 'max']

      self.mnt_best = inf if self.mnt_mode == 'min' else -inf

      self.early_stop = cfg_trainer.get('early_stop', inf)

    self.start_epoch = 0
    self.epoch = 0
    self.n_samples = 0
    self.n_steps = 0

    self.writer = SummaryWriter(config.log_dir)

    self.include_optim_in_ckpts = config['trainer'].get(
        'include_optim_in_ckpts', False)

    if config.resume is not None:
      self._resume_checkpoint(config.resume)

  @abc.abstractmethod
  def _train_epoch(self, epoch):
    """Training logic for an epoch."""
    raise NotImplementedError

  @abc.abstractmethod
  def _valid_epoch(self, epoch, sets):
    """Validation logic for an epoch."""
    raise NotImplementedError

  def train(self):
    """Full training logic."""
    not_improved_count = 0
    for epoch in range(self.start_epoch, self.epochs + 1):

      self.epoch = epoch
      epoch_start = time.time()

      logger.debug('Starting training epoch %s ...', str(epoch))
      train_start = time.time()
      result = self._train_epoch(epoch)
      for key, val in result.items():
        self.writer.add_scalar(f'{key}', val, epoch)
      self.timer.update('epoch.train', time.time() - train_start)

      logger.debug('Starting evaluating epoch %s ...', str(epoch))
      valid_start = time.time()
      val_log = self._valid_epoch(epoch, sets='continuous_eval')
      logger.debug('Updating val log with results ...')
      result.update(val_log)
      self.timer.update('epoch.valid', time.time() - valid_start)

      checkpoint_start = time.time()
      # save logged informations into log dict
      log = {'epoch': epoch}
      for key, value in result.items():
        # Metrics recorded during the continuous eval
        if key == 'metrics':
          for dataset_name, dataset_metrics in value.items():
            for metric_type, metric_dict in dataset_metrics.items():
              for metric_name, metric_value in metric_dict.items():
                log[f'{dataset_name}/{metric_type}'
                    f'/{metric_name}'] = metric_value
        else:
          log[key] = value

      # eval model according to configured metric, save best # ckpt as
      # trained_model.
      best = False
      if self.mnt_mode in ['min', 'max']:
        try:
          # check whether specified metric improved or not, according to
          # specified metric(mnt_metric)
          lower = log[self.mnt_metric] <= self.mnt_best
          higher = log[self.mnt_metric] >= self.mnt_best
          improved = (self.mnt_mode == 'min' and lower) or \
                     (self.mnt_mode == 'max' and higher)
        except KeyError:
          logger.warning(
              'Warning: Metric %s not found, '
              'perf monitoring is disabled.', self.mnt_metric)
          self.mnt_mode = 'off'
          improved = False
          not_improved_count = 0

        if improved:
          self.mnt_best = log[self.mnt_metric]
          not_improved_count = 0
          best = True
        else:
          not_improved_count += 1

        if not_improved_count > self.early_stop:
          logger.info(
              'Val performance didn\'t improve for %s epochs. '
              'Training stops.', self.early_stop)
          break

      # If checkpointing is done intermittently, still save models that
      # outperform the best metric.
      save_best = best and self.mnt_metric != 'epoch'

      if self.mnt_mode in ['given_epoch'] and epoch == self.given_epoch:
        save_best = True

      # Due to the fast runtime/slow HDD combination, checkpointing can dominate
      # the total training time, so we optionally skip checkpoints for some of
      # the first epochs
      if epoch < self.skip_first_n_saves:
        msg = f'Skipping ckpt save at epoch {epoch} < {self.skip_first_n_saves}'
        logger.info(msg)
      elif epoch % self.save_period == 0 or save_best:
        self._save_checkpoint(epoch, save_best=best)

      if epoch > self.num_keep_ckpts:
        self.purge_stale_checkpoints()
      self.timer.update('epoch.checkpoint', time.time() - checkpoint_start)

      self.timer.update('epoch.total', time.time() - epoch_start)
      for key, val in self.timer.dic.items():
        for metric in ['avg', 'sum']:
          log[f'timer.{key}.{metric}'] = self.timer.dic[key][metric]
        self.writer.add_scalar(f'timer_epoch/{key}', self.timer.dic[key]['sum'],
                               epoch)
      self.writer.add_text('exp_dir', str(self.exp_dir), epoch)
      self.timer.reset()

      log['mnt_best'] = self.mnt_best
      log['not_improved_count'] = not_improved_count
      self.writer.add_scalar('mnt_best', self.mnt_best, epoch)

      # print results
      for metric_name, metric_value in log.items():
        if '/cols' in metric_name:
          continue
        if 'timer.' in metric_name:
          logger.debug(' {:15s}: {}'.format(str(metric_name), metric_value))
        else:
          logger.info(' {:15s}: {}'.format(str(metric_name), metric_value))

      # Save main results in the perf log
      log_light = {}
      for key, value in log.items():
        if not key.endswith('cols'):
          log_light[key] = value
      update_perf_log(log_light, self.perf_log_path)

      # Log results to Tensorboard
      self.writer.add_hparams(self.hparams, {
          'hparam/accuracy': log[self.mnt_metric],
          'hparam/mnt_best': self.mnt_best,
          'hparam/epoch': epoch
      },
                              name='hparams')

      # # Ray-tune recording
      # try:
      #   from ray.tune import track
      #   acc = log[self.mnt_metric]
      #   track.log(mean_accuracy=acc, exp_dir=self.exp_dir, **log_light)
      # except Exception as e:
      #   print(e)

  def evaluate(self):
    """Final evaluation."""
    sets = 'final_eval'
    ckpt_path = self.config.save_dir / 'trained_model.pth'

    if os.path.exists(ckpt_path):
      self._resume_checkpoint(ckpt_path)
    else:
      msg = (f'The checkpoint {ckpt_path} does not exist and cannot be loaded. '
             f'The model will not be resumed to that checkpoint.')
      logger.info(msg)

    final_result = self._valid_epoch(epoch=self.epoch, sets=sets)
    nested_metrics = final_result['metrics']

    log = {}
    for dataset_name, dataset_metrics in nested_metrics.items():
      log[dataset_name] = {}
      for metric_type, metric_dict in dataset_metrics.items():
        for metric_name, metric_value in metric_dict.items():
          log[dataset_name][
              f'{metric_type}/{metric_name}/{sets}'] = metric_value

    # Print results
    for dataset_name, metric_dict in log.items():
      logger.info('%s:', dataset_name)
      for metric_name, metric_value in metric_dict.items():
        if '/cols' in metric_name:
          continue
        if 'timer.' in metric_name:
          logger.debug(' {:15s}: {}'.format(str(metric_name), metric_value))
        else:
          logger.info(' {:15s}: {}'.format(str(metric_name), metric_value))

    # Logging dataset perfs
    save_dir = self.config.save_dir
    results_on_datasets_log_path = os.path.join(save_dir, 'exp_results.json')
    if os.path.exists(results_on_datasets_log_path):
      with open(results_on_datasets_log_path) as json_file:
        res = json.load(json_file)
    else:
      res = collections.OrderedDict({})
    if 'perfs' not in res.keys():
      res['perfs'] = {}
    res['perfs'] = log
    res['checkpoint_epoch'] = self.loaded_epoch
    logger.info('Best epoch for the monitored metric: %s', self.loaded_epoch)
    with open(results_on_datasets_log_path, 'w') as fp:
      json.dump(res, fp, indent=4)

    exp_completed_flag_path = os.path.join(save_dir, 'exp_completed_flag.txt')
    # Touch the exp_completed_flag_path to mark that the experiment is completed
    with open(exp_completed_flag_path, 'a'):
      os.utime(exp_completed_flag_path, None)

  def purge_stale_checkpoints(self):
    """Remove checkpoints that are no longer neededself.

    NOTE: This function assumes that the `best` checkpoint has already been
    renamed
    to have a format that differs from `checkpoint-epoch<num>.pth`
    """
    found_epoch_ckpts = list(self.checkpoint_dir.glob('checkpoint-epoch*.pth'))
    if len(found_epoch_ckpts) <= self.num_keep_ckpts:
      return

    # purge the oldest checkpoints
    regex = r'.*checkpoint-epoch(\d+)[.]pth$'
    epochs = [
        int(re.search(regex, str(x)).groups()[0]) for x in found_epoch_ckpts
    ]
    sorted_ckpts = sorted(list(zip(epochs, found_epoch_ckpts)),
                          key=lambda x: -x[0])

    for epoch, stale_ckpt in sorted_ckpts[self.num_keep_ckpts:]:
      tic = time.time()
      stale_ckpt.unlink()
      msg = (f'removing stale ckpt [epoch {epoch}] '
             f'[took {time.time() - tic:.2f}s]')
      logger.info(msg)

  def _prepare_device(self, n_gpu_use):
    """Setup GPU device if available, move model into configured device."""
    n_gpu = torch.cuda.device_count()
    msg = f'n_gpu = torch.cuda.device_count(): {n_gpu} (nb of gpus available)'
    logger.debug(msg)
    if n_gpu_use > 0 and n_gpu == 0:
      logger.warning('Warning: There\'s no GPU available on this machine,'
                     'training will be performed on CPU.')
      n_gpu_use = 0
    if n_gpu_use > n_gpu:
      msg = ('Warning: The number of GPU\'s configured to use is {}'
             ', but only {} are available '
             'on this machine.'.format(n_gpu_use, n_gpu))
      logger.warning(msg)
      n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    logger.debug('device: %s', device)
    list_ids = list(range(n_gpu_use))
    logger.debug('list_ids: %s', list_ids)
    return device, list_ids

  def _save_checkpoint(self, epoch, save_best=False):
    """Saving checkpoints."""
    arch = type(self.model).__name__

    # To accomodate the DataParallel model that adds the prefix "module"
    # to the parameters
    try:
      state_dict = self.model.module.state_dict()
    except AttributeError:
      state_dict = self.model.state_dict()

    state = {
        'arch': arch,
        'epoch': epoch,
        'state_dict': state_dict,
        'monitor_best': self.mnt_best,
        'config': self.config,
        'n_samples': self.n_samples,
        'n_steps': self.n_steps,
    }
    if self.include_optim_in_ckpts:
      state['optimizer'] = self.optimizer.state_dict()
      state['lr_scheduler'] = self.lr_scheduler.state_dict()

    filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
    filename_tmp = filename + '_'
    tic = time.time()
    logger.info('Saving checkpoint: %s ...', filename)
    torch.save(state, filename_tmp)
    os.rename(filename_tmp, filename)
    msg = f'Done in {time.time() - tic:.3f}s'
    logger.info(msg)
    if save_best:
      logger.info('Updating \'best\' checkpoint: %s ...', filename)
      best_path = str(self.checkpoint_dir / 'trained_model.pth')
      best_path_tmp = best_path + '_'
      torch.save(state, best_path_tmp)
      os.rename(best_path_tmp, best_path)
      msg = f'Done in {time.time() - tic:.3f}s'
      logger.info(msg)

  def _resume_last_checkpoint(self):
    checkpoint_path = get_last_checkpoint_path(self.exp_dir)
    self._resume_checkpoint(checkpoint_path)

  def match_checkpoint_to_model(self, checkpoint, model):
    """Adapt the loaded checkpoint so that is fits the current architecture."""

    modules = ['vid_bert.embeddings.position_embeddings.weight']

    for module in modules:
      if module in model and checkpoint[module].shape != model[module].shape:
        padding = model[module].shape[0] - checkpoint[module].shape[0]
        padding_shape = list(model[module].shape)
        padding_shape[0] = padding
        device = checkpoint[module].device
        checkpoint[module] = torch.cat(
            [checkpoint[module],
             torch.zeros(padding_shape, device=device)], 0)
        logger.warning('Size mismatch for module %s fixed by zero padding',
                       module)

  def _resume_checkpoint(self, resume_path):
    """Resume from saved checkpoints."""
    resume_path = str(resume_path)
    logger.info('Loading checkpoint from: %s ...', resume_path)
    checkpoint = torch.load(resume_path, map_location=self.device)
    self.loaded_epoch = checkpoint['epoch']
    self.epoch = checkpoint['epoch']
    self.start_epoch = checkpoint['epoch'] + 1
    self.n_samples = checkpoint['n_samples']
    self.n_steps = checkpoint['n_steps']

    exp_dir_src = os.path.dirname(resume_path)
    restart = exp_dir_src == str(self.exp_dir)

    # load architecture params from checkpoint.
    if checkpoint['config']['arch'] != self.config['arch']:
      msg = ('Warning: Architecture configuration given in config file is'
             'different from that of checkpoint. This may yield an exception'
             ' while state_dict is being loaded.')
      logger.warning(msg)
      logger.warning('Created model conf: %s', self.config['arch'])
      logger.warning('Loaded model conf: %s', checkpoint['config']['arch'])
    self.match_checkpoint_to_model(checkpoint['state_dict'],
                                   self.model.state_dict())
    self.model.load_state_dict(checkpoint['state_dict'], strict=restart)

    if restart:
      # load optimizer state from ckpt only when optimizer type is not changed.
      optim_args = checkpoint['config']['optimizer']
      if optim_args['type'] != self.config['optimizer']['type']:
        msg = ('Warning: Optimizer type given in config file differs from that'
               ' of checkpoint. Optimizer parameters not being resumed.')
        logger.warning(msg)
      else:
        self.optimizer.load_state_dict(checkpoint['optimizer'])
      lr_scheduler_args = checkpoint['config']['lr_scheduler']
      if lr_scheduler_args['type'] != self.config['lr_scheduler']['type']:
        msg = (
            'Warning: Lr_scheduler type given in config file differs from that'
            ' of checkpoint. Lr_scheduler parameters not being resumed.')
        logger.warning(msg)
      else:
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
      self.mnt_best = checkpoint['monitor_best']
    else:
      self.loaded_epoch = 0
      self.epoch = 0
      self.start_epoch = 0
      self.n_samples = 0
      self.n_steps = 0

      # Log the path of the checkpoint that was loaded
      with open(self.info_checkpoint_path, 'a') as f:
        f.write(f"This experiment is based on the checkpoint {resume_path}"
                f"loaded at epoch {checkpoint['epoch']}")

    logger.info('Ckpt loaded at epoch %s.', str(checkpoint['epoch']))
