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

import collections
import logging
import pathlib
import time

from base import BaseTrainer
from model.model import sharded_cross_view_inner_product
import numpy as np
import pytorch_warmup as warmup
import torch
from utils.util import compress_predictions

logger = logging.getLogger(__name__)


def move_dict_to_device(res, device, only_tensors=True):
  """Move a dictionnary to another device memory."""
  for key in list(res.keys()):
    value = res[key]
    if isinstance(value, np.ndarray):
      res[key] = torch.from_numpy(res[key])
      if device is not None:
        res[key] = res[key].to(device)
    elif isinstance(value, torch.Tensor):
      if device is not None:
        res[key] = value.to(device)
    elif isinstance(value, collections.OrderedDict) or isinstance(value, dict):
      res[key] = move_dict_to_device(res[key], device)
    else:
      if only_tensors:
        res.pop(key)
  return res


class Trainer(BaseTrainer):
  """Trainer class.

  Note:
      Inherited from BaseTrainer.
  """

  def __init__(self,
               model,
               loss,
               metrics,
               optimizer,
               config,
               data_loaders,
               lr_scheduler,
               visualizer,
               skip_first_n_saves,
               include_optim_in_ckpts,
               expert_dims,
               num_keep_ckpts=1,
               tokenizer=None,
               warmup_iterations=-1):
    super().__init__(model, loss, metrics, optimizer, lr_scheduler, config)
    self.config = config
    self.data_loaders = data_loaders
    self.num_keep_ckpts = num_keep_ckpts
    self.train_loaders = [
        lo["loader"] for lo in self.data_loaders["train_sets"]
    ]
    self.train_datasets = [
        lo["dataset"] for lo in self.data_loaders["train_sets"]
    ]
    self.continuous_eval_loaders = [
        lo["loader"] for lo in self.data_loaders["continuous_eval_sets"]
    ]
    self.continuous_eval_datasets = [
        lo["dataset"] for lo in self.data_loaders["continuous_eval_sets"]
    ]
    self.log_step = int(np.sqrt(self.train_loaders[0].batch_size))
    self.visualizer = visualizer
    self.skip_first_n_saves = skip_first_n_saves
    self.include_optim_in_ckpts = include_optim_in_ckpts

    self.batch_size = self.train_loaders[0].batch_size
    self.n_pairs = self.train_datasets[0].n_pairs

    self.modalities = list(expert_dims.keys())

    cfg_trainer = config["trainer"]
    self.max_samples_per_epoch = cfg_trainer["max_samples_per_epoch"]
    self.max_batches_per_epoch = int(self.max_samples_per_epoch / self.n_pairs
                                     / self.batch_size)
    self.batches_per_epoch = min(len(self.train_loaders[0]),
                                 self.max_batches_per_epoch)
    self.samples_per_epoch = self.batches_per_epoch * self.batch_size * self.n_pairs

    self.debug_dataloader = False
    self.tokenizer = tokenizer

    if warmup_iterations > 0:
      self.warmup_scheduler = warmup.LinearWarmup(
          optimizer, warmup_period=warmup_iterations)
    else:
      self.warmup_scheduler = None

  def _train_epoch(self, epoch):
    # There is no training at epoch 0, only evaluation
    if epoch == 0:
      logger.debug("Performing initial evaluation...")
      total_loss = 0
      log = {"loss": total_loss}
      learning_rate = self.lr_scheduler.get_lr()[0]
      log["learning_rate"] = learning_rate
      log["n_samples"] = self.n_samples
      log["n_steps"] = self.n_steps
      return log

    self.model.train()
    total_loss = 0
    out = "embds" if isinstance(self.model, torch.nn.DataParallel) else "conf"

    # Choose which of the trainsets to use (pretraining or finetuning)
    i = 0
    while self.data_loaders["train_sets"][i].until_epoch < epoch:
      i += 1

    self.batch_size = self.data_loaders["train_sets"][i].batch_size
    self.n_pairs = self.data_loaders["train_sets"][i].n_pairs
    self.source = self.data_loaders["train_sets"][i]["dataset"].dataset_name

    msg = (f"Out: {out}, source: {self.source}, batch_size: {self.batch_size}, "
           f"n_pairs: {self.n_pairs}")
    logger.debug(msg)

    data_start = time.time()
    for batch_idx, minibatch in enumerate(self.train_loaders[i]):
      # We limit the number of samples per epoch
      if (batch_idx
          + 1) * self.batch_size * self.n_pairs > self.max_samples_per_epoch:
        break

      self.timer.update("train_batch.data_loading", time.time() - data_start)
      dataset_name = self.train_loaders[0].dataset.dataset_name

      transfer_start = time.time()
      # Print the minibatch data to debug the dataloader
      if self.debug_dataloader and batch_idx < 1:
        self.display_minibatch(minibatch, dataset_name)
        debug = True
      else:
        debug = False

      minibatch = move_dict_to_device(minibatch, self.device)

      self.n_samples += self.batch_size * self.n_pairs
      self.n_steps += 1

      if self.warmup_scheduler:
        self.warmup_scheduler.dampen()

      self.optimizer.zero_grad()
      self.timer.update("train_batch.transfer", time.time() - transfer_start)
      forward_start = time.time()
      output = self.model(**minibatch, out=out, device=self.device, debug=debug)

      self.timer.update("train_batch.forward", time.time() - forward_start)
      loss_start = time.time()
      if out == "conf":
        loss = self.loss(output["cross_view_conf_matrix"])
      else:
        # Reassemble the tensors from multiple GPUs to one dict
        vid_embds = collections.OrderedDict()
        text_embds = collections.OrderedDict()
        for idx, mod in enumerate(self.modalities):
          vid_embds[mod] = output["vid_embds"][:, idx]
          text_embds[mod] = output["text_embds"][:, idx]
        cross_view_conf_matrix = sharded_cross_view_inner_product(
            vid_embds=vid_embds,
            text_embds=text_embds,
            vid_weights=output["vid_weights"],
            text_weights=output["text_weights"],
            subspaces=self.modalities,
            merge_caption_similiarities="avg",
        )
        loss = self.loss(cross_view_conf_matrix)

      self.timer.update("train_batch.loss", time.time() - loss_start)
      backward_start = time.time()
      loss.backward()
      self.optimizer.step()

      loss_value = loss.item()
      total_loss += loss_value

      self.timer.update("train_batch.backward", time.time() - backward_start)
      self.timer.update("train_batch.total", time.time() - data_start)

      logging_start = time.time()
      display_timers = False
      if batch_idx % self.log_step == 0:
        prog = self._progress(batch_idx)
        data_loading_time = self.timer.dic["train_batch.data_loading"]["val"]
        forward_time = self.timer.dic["train_batch.forward"]["val"]
        loss_time = self.timer.dic["train_batch.loss"]["val"]
        backward_time = self.timer.dic["train_batch.backward"]["val"]
        batch_time = self.timer.dic["train_batch.total"]["val"]
        msg = (f"Train Epoch: {epoch} {prog} Loss: {loss_value:.6f} "
               f"batch_time={batch_time:.5f} ")
        if display_timers:
          msg = msg + (f"data_loading={data_loading_time:.5f}"
                       f"({data_loading_time * 100 / batch_time:.0f}%) "
                       f"forward={forward_time:.5f}"
                       f"({forward_time * 100 / batch_time:.0f}%) "
                       f"loss={loss_time:.5f}"
                       f"({loss_time * 100 / batch_time:.0f}%) "
                       f"backward={backward_time:.5f}"
                       f"({backward_time * 100 / batch_time:.0f}%) ")
        logger.info(msg)

      self.timer.update("train.logging", time.time() - logging_start)
      data_start = time.time()

    scheduler_start = time.time()
    log = {"loss": total_loss / self.batches_per_epoch}

    learning_rate = self.lr_scheduler.get_lr()[0]
    log["learning_rate"] = learning_rate
    log["n_samples"] = self.n_samples
    log["n_steps"] = self.n_steps

    if self.lr_scheduler is not None:
      self.lr_scheduler.step()

    self.timer.update("train_batch.scheduler", time.time() - scheduler_start)
    return log

  def display_minibatch(self, minibatch, dataset_name):
    b = len(minibatch["token_ids"])
    for i in range(b):
      logger.debug()
      msg = (f"Sample {i}: dataset: {dataset_name}, "
             f"source: {minibatch['sources'][i]}")
      logger.debug(msg)


#       if "raw_captions" in minibatch.keys():
#         logger.debug(f"raw_captions: {minibatch['raw_captions'][i]}")
#         logger.debug(f"raw_captions_t: {minibatch['raw_captions_t'][i]}")
#         logger.debug(f"paths: {minibatch['paths'][i]}")
#       logger.debug(f"query_masks: {minibatch['query_masks'][i]}")
#       ids = minibatch["token_ids"][i, 0, :, 0]
#       logger.debug(f"token_ids: {ids}")
#       inds = minibatch["token_ids"][i, 0, :, 1]
#       logger.debug(f"token_inds: {inds}")

#       tokens = self.tokenizer.convert_ids_to_tokens(ids)
#       logger.debug(f"tokens: {tokens}")

#       for key, val in minibatch["features"].items():
#         logger.debug(f"features: {key}: {val[i]}")
#         val_t = minibatch["features_t"][key]
#         logger.debug(f"features_t: {key}: {val_t[i]}")
#         val_ind = minibatch["features_ind"][key]
#         logger.debug(f"features_ind: {key}: {val_ind[i]}")

  def log_metrics(self, metric_store, epoch, metric_name, dataset_name):
    for key, value in metric_store.items():
      if key != "cols":
        self.writer.add_scalar(f"{dataset_name}/{metric_name}/{key}", value,
                               epoch)

  def _get_embeddings(self, val_loader):
    out = "embds"
    with torch.no_grad():
      vid_embds = collections.OrderedDict()
      text_embds = collections.OrderedDict()
      query_masks_list = []
      raw_captions_list = []
      token_ids_list = []
      paths_list = []
      vid_weights_list = []
      text_weights_list = []
      logger.debug("Running batches through model ...")
      data_start = time.time()
      for batch_idx, minibatch in enumerate(val_loader):
        self.timer.update("valid_batch.data_loading", time.time() - data_start)

        transfer_start = time.time()
        if "raw_captions" in minibatch.keys():
          raw_captions_list.extend(minibatch["raw_captions"])
          paths_list.extend(minibatch["paths"])
        else:
          b, captions_per_video, _, _ = minibatch["token_ids"].size()
          raw_caps = [[np.array(["unprovided", "words"])] * captions_per_video
                     ] * b
          raw_captions_list.extend(raw_caps)
          paths = [pathlib.Path("unprovided/path")] * b
          paths_list.extend(paths)

        if "token_ids" in minibatch.keys():
          token_ids_list.extend(minibatch["token_ids"])

        query_masks_list.append(torch.from_numpy(minibatch["query_masks"]))

        # Print the minibatch data to debug the dataloader
        if self.debug_dataloader and batch_idx == 0:
          dataset_name = val_loader.dataset.dataset_name
          self.display_minibatch(minibatch, dataset_name)
          debug = True
        else:
          debug = False

        minibatch = move_dict_to_device(minibatch, self.device)

        self.timer.update("valid_batch.transfer", time.time() - transfer_start)

        forward_start = time.time()

        output = self.model(**minibatch,
                            out=out,
                            device=self.device,
                            debug=debug)

        vid_weights_list.append(output["vid_weights"])
        text_weights_list.append(output["text_weights"])
        for idx, mod in enumerate(self.modalities):
          vid_embds.setdefault(mod, []).append(output["vid_embds"][:, idx])
          text_embds.setdefault(mod, []).append(output["text_embds"][:, idx])
        self.timer.update("valid_batch.forward", time.time() - forward_start)
        self.timer.update("valid_batch.total", time.time() - data_start)

        data_start = time.time()

      query_masks = torch.cat(query_masks_list, 0)
      vid_weights = torch.cat(vid_weights_list, 0)
      text_weights = torch.cat(text_weights_list, 0)
      for idx, mod in enumerate(self.modalities):
        vid_embds[mod] = torch.cat(vid_embds[mod], 0)
        text_embds[mod] = torch.cat(text_embds[mod], 0)

      token_ids = np.concatenate(token_ids_list)

      res = {
          "vid_embds": vid_embds,
          "text_embds": text_embds,
          "vid_weights": vid_weights,
          "text_weights": text_weights,
          "raw_captions": raw_captions_list,
          "token_ids": token_ids,
          "query_masks": query_masks,
          "paths": paths_list,
      }

      move_dict_to_device(res, device="cpu", only_tensors=False)

      return res

  def _valid_epoch(self, epoch=None, sets="continuous_eval"):
    embds_start = time.time()
    loaders = [lo["loader"] for lo in self.data_loaders[f"{sets}_sets"]]
    datasets = [lo["dataset"] for lo in self.data_loaders[f"{sets}_sets"]]

    self.model.eval()

    result_dict = {}
    result_dict["metrics"] = {}

    loaders_embds = {}

    with torch.no_grad():
      for loader_idx, val_loader in enumerate(loaders):
        dataset = datasets[loader_idx]
        split_name = dataset.split_name
        dataset_name = dataset.dataset_name
        logger.debug("Obtaining embeddings for dataset %s ...", dataset_name)
        loaders_embds[dataset_name] = self._get_embeddings(val_loader)

      self.timer.update("valid.embds", time.time() - embds_start)
      for dataset_name, embds in loaders_embds.items():
        conf_mat_start = time.time()
        logger.debug("Computing confusion matrix ...")
        cross_view_conf_matrix = sharded_cross_view_inner_product(
            vid_embds=embds["vid_embds"],
            text_embds=embds["text_embds"],
            vid_weights=embds["vid_weights"],
            text_weights=embds["text_weights"],
            subspaces=self.modalities,
            merge_caption_similiarities="indep",
        )
        sims = cross_view_conf_matrix.data.cpu().float().numpy()
        query_masks = embds["query_masks"].numpy()

        dataset_basename = dataset_name.split("_")[0]
        cut_name = dataset_name.split("_")[1]
        split_name = dataset_name.split("_")[2]

        if sets == "final_eval":
          # Challenge mode, we record the rankings
          if cut_name == "c" and split_name in ["test1", "test2"]:
            # example: MSRVTT-public_server_val-predictions.csv
            if split_name in ["test1"]:
              split_name = "public_server_val"
            elif split_name in ["test2"]:
              split_name = "public_server_test"
            prediction_path = pathlib.Path(
                self.exp_dir
            ) / f"{dataset_basename}-{split_name}-predictions.csv"
            compressed_preds = compress_predictions(
                query_masks=query_masks,
                sims=sims,
            )
            np.savetxt(prediction_path,
                       compressed_preds,
                       delimiter=",",
                       fmt="%d")
            logger.debug("Saved similarity matrix predictions to %s",
                         prediction_path)

          sims_data = {"sims": sims, "query_masks": query_masks}
          sims_path = pathlib.Path(
              self.exp_dir) / f"{dataset_basename}-{split_name}-sims.npy"
          np.save(sims_path, sims_data)
          logger.info("Saved similarity matrix to %s", str(sims_path))

        nested_metrics = {}
        self.timer.update("valid.conf_mat", time.time() - conf_mat_start)
        metrics_start = time.time()

        logger.debug("Computing metrics ...")
        for metric in self.metrics:
          metric_name = metric.__name__
          logger.debug("Computing %s metric ...", metric_name)
          nested_metrics[metric_name] = metric(sims, query_masks=query_masks)

          # Log the metrics in Tensorboard
          logger.debug("Logging %s on Tensorboard ...", metric_name)
          self.log_metrics(metric_store=nested_metrics[metric_name],
                           epoch=epoch,
                           metric_name=metric_name,
                           dataset_name=dataset_name)

        result_dict["metrics"][dataset_name] = nested_metrics
        self.timer.update("valid.metrics", time.time() - metrics_start)

        # Qualitative evaluation of the rankings
        visu_start = time.time()
        meta = {}
        meta["paths"] = embds["paths"]
        meta["raw_captions"] = embds["raw_captions"]
        meta["vid_weights"] = embds["vid_weights"]
        meta["text_weights"] = embds["text_weights"]
        meta["token_ids"] = embds["token_ids"]
        subdir_name = f"{split_name}_{sets}"
        if meta["raw_captions"] is not None:
          logger.debug("Visualizing the ranking ...")
          self.visualizer.visualize_ranking(
              sims=sims,
              query_masks=query_masks,
              epoch=epoch,
              meta=meta,
              nested_metrics=nested_metrics,
              modalities=self.modalities,
              subdir_name=subdir_name,
              sets=sets,
              tokenizer=self.tokenizer,
          )
        self.timer.update("valid.visu", time.time() - visu_start)

    return result_dict

  def _progress(self, batch_idx):
    base = "[{}/{} {}/{} ({:.0f}%)]"
    current = batch_idx + 1
    total = self.batches_per_epoch
    current_samples = (batch_idx + 1) * self.batch_size * self.n_pairs
    total_samples = self.samples_per_epoch
    return base.format(current, total, current_samples, total_samples,
                       100.0 * current / total)
