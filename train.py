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
"""Cross-modal architecture training.

Code based on the implementation of "Collaborative Experts":
https://github.com/albanie/collaborative-experts
"""

import argparse
import logging
import os
import random
import time

import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
import numpy as np
from parse_config import ConfigParser
import torch
from trainer import Trainer
from utils import ranger
from utils.nlp_utils import create_tokenizer
from utils.util import compute_dims
import utils.visualizer as module_vis

logger = logging.getLogger(__name__)


def train(config):
  """Cross-modal architecture training."""

  # Get the list of experts and their dimensions
  expert_dims = compute_dims(config)
  raw_input_dims = {}
  for expert, expert_dic in expert_dims.items():
    raw_input_dims[expert] = expert_dic["dim"]

  # Set the random initial seeds
  tic = time.time()
  seed = config["seed"]
  cross_seed = config.get("cross_seed", seed)
  logger.debug("Setting experiment random seed to %d", seed)
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)

  # Tokenizer to parse sentences into tokens
  tokenizer = create_tokenizer(config["arch"]["args"]["txt_inp"])

  # Create the datasets
  logger.info("Preparing the dataloaders ...")
  dataset_types = ["train_sets", "continuous_eval_sets", "final_eval_sets"]
  data_loaders = {}
  loaded_data = {}
  for dataset_type in dataset_types:
    training = dataset_type == "train_sets"
    if not config.get(dataset_type, False):
      continue
    data_loaders[dataset_type] = []
    for _, data_loader in enumerate(config[dataset_type]):
      data_loaders[dataset_type].append(
          getattr(module_data, data_loader["type"])(
              **data_loader["args"],
              raw_input_dims=raw_input_dims,
              training=training,
              tokenizer=tokenizer,
              loaded_data=loaded_data,
              cross_seed=cross_seed,
          ))

  # Setup the cross-modal architecture
  model = config.init(
      name="arch",
      module=module_arch,
      expert_dims=expert_dims,
      tokenizer=tokenizer,
  )

  loss = config.init(name="loss", module=module_loss)
  metrics = [getattr(module_metric, met) for met in config["metrics"]]
  trainable_params = filter(lambda p: p.requires_grad, model.parameters())

  if config["optimizer"]["type"] == "Ranger":
    optimizer = config.init("optimizer", ranger, trainable_params)
  else:
    optimizer = config.init("optimizer", torch.optim, trainable_params)

  lr_scheduler = config.init("lr_scheduler", torch.optim.lr_scheduler,
                             optimizer)

  if "warmup_iterations" in config["optimizer"]:
    warmup_iterations = config["optimizer"]["warmup_iterations"]
  else:
    warmup_iterations = -1

  visualizer = config.init(
      name="visualizer",
      module=module_vis,
      exp_name=config.exper_name,
      web_dirs=config.web_dirs,
  )

  trainer = Trainer(
      model,
      loss,
      metrics,
      optimizer,
      config=config,
      data_loaders=data_loaders,
      lr_scheduler=lr_scheduler,
      visualizer=visualizer,
      skip_first_n_saves=config["trainer"].get("skip_first_n_saves", 0),
      include_optim_in_ckpts=config["trainer"].get("include_optim_in_ckpts",
                                                   False),
      expert_dims=expert_dims,
      tokenizer=tokenizer,
      warmup_iterations=warmup_iterations)

  if not config.only_eval:
    logger.info("Training ...")
    trainer.train()
  logger.info("Final evaluation ...")
  trainer.evaluate()
  duration = time.strftime("%Hh%Mm%Ss", time.gmtime(time.time() - tic))
  logger.info("Script took %s", duration)

  # Report the location of the "best" checkpoint of the final seeded run (here
  # "best" corresponds to the model with the highest geometric mean over the
  # R@1, R@5 and R@10 metrics when a validation set is used, or simply the final
  # epoch of training for fixed-length schedules).
  best_ckpt_path = config.save_dir / "trained_model.pth"
  if os.path.exists(best_ckpt_path):
    logger.info("The best performing ckpt can be found at %s",
                str(best_ckpt_path))


def main_train(raw_args=None):
  parser = argparse.ArgumentParser(description="PyTorch Template")
  parser.add_argument("--config",
                      default=None,
                      type=str,
                      help="config file path (default: None)")
  parser.add_argument(
      "--resume",
      default=None,
      type=str,
      help="path to the experiment dir to resume (default: None)")
  parser.add_argument("--load_checkpoint",
                      default=None,
                      type=str,
                      help="path to the checkpoint to load (default: None)")
  parser.add_argument("--device", type=str, help="indices of GPUs to enable")
  parser.add_argument("--only_eval", action="store_true")
  parser.add_argument("-v",
                      "--verbose",
                      help="increase output verbosity",
                      action="store_true")
  args = parser.parse_args(raw_args)
  args = ConfigParser(args)

  msg = (
      f"Expected the number of training epochs ({args['trainer']['epochs']})"
      f"to exceed the save period ({args['trainer']['save_period']}), otherwise"
      " no checkpoints will be saved.")
  assert args["trainer"]["epochs"] >= args["trainer"]["save_period"], msg

  train(config=args)


if __name__ == "__main__":
  main_train()
