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
"""HowTo100M dataset."""
import os
import time
from base.base_dataset import BaseDataset


class HowTo100M(BaseDataset):
  """HowTo100M dataset."""

  def configure_train_test_splits(self, cut_name, split_name):
    self.restrict_test_captions = None
    list_path = None
    if cut_name in ["full"]:
      if split_name in ["train"]:
        list_path = "train_list_full.txt"
      elif split_name in ["trn"]:
        list_path = "trn_list_full.txt"
      elif split_name in ["val", "valong", "val3-30"]:
        list_path = "val_list_full.txt"
      elif split_name in ["test", "testlong", "test3-30"]:
        list_path = "test_list_full.txt"
    else:
      msg = "unrecognised HowTo100M cut: {}"
      raise ValueError(msg.format(cut_name))

    list_path = os.path.join(self.root_feat, list_path)

    print("loading training/val splits....")
    tic = time.time()
    with open(list_path) as f:
      self.vid_list = f.readlines()
    self.vid_list = [x.strip() for x in self.vid_list]
    print("done in {:.3f}s".format(time.time() - tic))

    self.split_name = split_name
    self.dataset_name = f"HowTo100M_{cut_name}_{split_name}"
