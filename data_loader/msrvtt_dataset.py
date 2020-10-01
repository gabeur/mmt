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
"""MSR-VTT dataset."""
import os
import os.path

from base.base_dataset import BaseDataset
import numpy as np
from utils.util import get_expert_paths
from utils.util import memcache


class MSRVTT(BaseDataset):
  """MSR-VTT dataset."""

  def configure_train_test_splits(self, cut_name, split_name):
    self.restrict_test_captions = None

    if cut_name in ["miech", "jsfusion"]:
      if cut_name in ["miech"]:
        # For now, we follow Antoine's approach of using the first text caption
        # for the retrieval task when evaluating on his custom split.
        train_list_path = "train_list_miech.txt"
        test_list_path = "test_list_miech.txt"
      elif cut_name in ["jsfusion"]:
        train_list_path = "train_list_jsfusion.txt"
        test_list_path = "val_list_jsfusion.txt"
        # NOTE: The JSFusion split (referred to as 1k-A in the paper) uses all
        # videos, but randomly samples a single caption per video from the test
        # set for evaluation. To reproduce this evaluation, we use the indices
        # of the test captions, and restrict to this subset during eval.
        test_cap_idx_path = os.path.join(self.data_dir,
                                         "jsfusion_val_caption_idx.pkl")
        self.restrict_test_captions = memcache(test_cap_idx_path)

      test_list_path = os.path.join(self.data_dir, test_list_path)
      with open(test_list_path) as f:
        test_vid_list = f.readlines()
      nb_test_samples = len(test_vid_list)

      if split_name in ["train", "trn", "val", "trainval"]:
        train_list_path = os.path.join(self.data_dir, train_list_path)
        with open(train_list_path) as f:
          train_vid_list = f.readlines()
        nb_train_samples = len(train_vid_list)

        cross_vid_list = train_vid_list
        cross_vid_list = [x.strip() for x in cross_vid_list]

        # The cross seed is used to split training videos into different
        # cross validation splits.
        rng = np.random.RandomState(self.cross_seed)
        rng.shuffle(cross_vid_list)

        if split_name in ["train", "trn", "trainval"]:
          if split_name in ["trainval"]:
            self.vid_list = cross_vid_list
          elif split_name in ["train", "trn"]:
            self.vid_list = cross_vid_list[nb_test_samples:]
          if split_name in ["trn"]:
            self.vid_list = self.vid_list[:nb_test_samples]

        elif split_name in ["val"]:
          self.vid_list = cross_vid_list[:nb_test_samples]

      elif split_name == "test":
        self.vid_list = test_vid_list
        self.vid_list = [x.strip() for x in self.vid_list]

    elif cut_name in ["full"]:
      if split_name in ["train", "trn"]:
        list_path = "train_list.txt"
      elif split_name in ["val"]:
        list_path = "val_list.txt"
      elif split_name in ["test"]:
        list_path = "test_list.txt"
      else:
        raise ValueError(f"unrecognised split: {split_name}")
      list_path = os.path.join(self.data_dir, list_path)
      with open(list_path) as f:
        self.vid_list = f.readlines()
      self.vid_list = [x.strip() for x in self.vid_list]

      # We want the trn split to be the same size as the val set
      if split_name in ["trn"]:
        rng = np.random.RandomState(0)
        rng.shuffle(self.vid_list)
        self.vid_list = self.vid_list[:497]

    elif cut_name in ["c"]:
      self.expert_paths = get_expert_paths(self.data_dir)
      if split_name in ["train", "trn", "val", "trainval"]:
        train_list_path = "train_list.txt"
        train_list_path = os.path.join(self.data_dir, train_list_path)
        with open(train_list_path) as f:
          train_vid_list = f.readlines()
        nb_train_samples = len(train_vid_list)

        val_list_path = "val_list.txt"
        val_list_path = os.path.join(self.data_dir, val_list_path)
        with open(val_list_path) as f:
          val_vid_list = f.readlines()
        nb_val_samples = len(val_vid_list)

        cross_vid_list = train_vid_list + val_vid_list
        cross_vid_list = [x.strip() for x in cross_vid_list]

        if self.cross_seed != 0:
          # The cross seed is used to split training videos into different
          # cross validation splits.
          rng = np.random.RandomState(self.cross_seed)
          rng.shuffle(cross_vid_list)

        if split_name in ["train", "trn", "trainval"]:
          if split_name in ["trainval"]:
            self.vid_list = cross_vid_list
          elif split_name in ["train", "trn"]:
            self.vid_list = cross_vid_list[:nb_train_samples]
          if split_name in ["trn"]:
            # In order to monitor performance on the training set, we sample
            # from it as many samples as there are validation samples.
            rng = np.random.RandomState(0)
            rng.shuffle(self.vid_list)
            self.vid_list = self.vid_list[:nb_val_samples]

        elif split_name in ["val"]:
          self.vid_list = cross_vid_list[nb_train_samples:]

      else:
        if split_name == "test1":
          list_path = "public_server_val.txt"
        elif split_name == "test2":
          list_path = "public_server_test.txt"
        list_path = os.path.join(self.data_dir, list_path)
        with open(list_path) as f:
          self.vid_list = f.readlines()
        self.vid_list = [x.strip() for x in self.vid_list]

    else:
      msg = "unrecognised cut: {}"
      raise ValueError(msg.format(cut_name))

    self.split_name = split_name
    self.dataset_name = f"MSRVTT_{cut_name}_{split_name}"
