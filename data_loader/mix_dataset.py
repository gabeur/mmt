# Copyright 2020 Valentin Gabeur
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
"""Logic for loading samples coming from different datasets."""
import abc
import logging

from data_loader.activitynet_dataset import ActivityNet
from data_loader.didemo_dataset import DiDeMo
from data_loader.howto100m_dataset import HowTo100M
from data_loader.lsmdc_dataset import LSMDC
from data_loader.msrvtt_dataset import MSRVTT
from data_loader.msvd_dataset import MSVD
from data_loader.youcook2_dataset import YouCook2
import numpy as np
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class MixDataset(Dataset):
  """Dataset composed of a mix of different datasets."""

  @abc.abstractmethod
  def configure_train_test_splits(self, split_name):
    """Partition the datset into train/val/test splits."""
    raise NotImplementedError

  @abc.abstractmethod
  def sanity_checks(self):
    """Run sanity checks on loaded data."""
    raise NotImplementedError

  @abc.abstractmethod
  def load_features(self):
    """Load features from disk."""
    raise NotImplementedError

  def __init__(self,
               mix,
               raw_input_dims,
               training=False,
               tokenizer=None,
               n_pairs=1,
               loaded_data=None,
               cross_seed=0):

    self.sanity_checks = False
    self.mix = mix
    self.experts = set(raw_input_dims.keys())
    self.train = training
    self.tokenizer = tokenizer
    self.n_pairs = n_pairs

    if len(mix) == 1:
      self.dataset_name = "_".join(
          [mix[0]["dataset_name"], mix[0]["cut_name"], mix[0]["split_name"]])
      self.split_name = mix[0]["split_name"]
    else:
      self.dataset_name = "Mix"
      self.split_name = "mic"

    dataset_classes = {
        "MSVD": MSVD,
        "LSMDC": LSMDC,
        "MSRVTT": MSRVTT,
        "DiDeMo": DiDeMo,
        "ActivityNet": ActivityNet,
        "YouCook2": YouCook2,
        "HowTo100M": HowTo100M,
    }

    self.datasets = []
    self.mix_weights = []
    self.dataset_names = []
    for config in mix:
      dataset_config = config.copy()
      if "mix_weight" in dataset_config.keys():
        self.mix_weights.append(dataset_config["mix_weight"])
        dataset_config.pop("mix_weight")
      else:
        self.mix_weights.append(1)

      dataset_name = dataset_config["dataset_name"]
      self.dataset_names.append(dataset_name)
      dataset_config.pop("dataset_name")
      dataset = dataset_classes[dataset_name](**dataset_config,
                                              raw_input_dims=raw_input_dims,
                                              training=training,
                                              tokenizer=tokenizer,
                                              n_pairs=n_pairs,
                                              loaded_data=loaded_data,
                                              cross_seed=cross_seed)
      self.datasets.append(dataset)

    self.mix_weights = [
        float(i) / sum(self.mix_weights) for i in self.mix_weights
    ]
    logger.debug("Datasets: %s", self.dataset_names)
    logger.debug("mix_weights: %s", self.mix_weights)

  def collate_data(self, data):
    text_keys = data[0]["text_tensors"].keys()
    text_tensors = {key: [] for key in text_keys}

    vid_keys = data[0]["vid_tensors"].keys()
    vid_tensors = {
        key: {expert: [] for expert in self.experts} for key in vid_keys
    }

    l_keys = data[0]["lists"].keys()
    lists = {key: [] for key in l_keys}

    for _, vid in enumerate(data):
      for key in text_keys:
        text_tensors[key].append(vid["text_tensors"][key])
      for key in vid_keys:
        for expert in self.experts:
          vid_tensors[key][expert].append(vid["vid_tensors"][key][expert])
      for key in l_keys:
        lists[key].extend(vid["lists"][key])

    # Concatenate the arrays of each sample to form a batch
    for key in text_keys:
      text_tensors[key] = np.concatenate(text_tensors[key],
                                         axis=0).astype(np.int32)
    for key in vid_keys:
      for expert in self.experts:
        vid_tensors[key][expert] = np.concatenate(vid_tensors[key][expert],
                                                  axis=0).astype(np.float32)

    minibatch = {**text_tensors, **vid_tensors, **lists}

    return minibatch

  def __len__(self):
    if len(self.mix) == 1:
      if self.train:
        # If it is a training dataset, we let the trainer decide when the epoch
        # is completed.
        return int(1E7)
      else:
        return len(self.datasets[0])
    else:
      if self.train:
        # If it is a training dataset, we let the trainer decide when the epoch
        # is completed.
        return int(1E7)
      else:
        # Normaly there should not be evaluation on the mix dataset
        return 1000

  def __getitem__(self, idx):

    if self.train:
      rng = np.random
    else:
      # Deterministic
      rng = np.random.RandomState(idx)

    # Select the dataset
    dataset_nb = rng.choice(len(self.mix), p=self.mix_weights)
    dataset = self.datasets[dataset_nb]

    return dataset[idx]
