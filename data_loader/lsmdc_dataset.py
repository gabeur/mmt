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
"""LSMDC dataset."""
import os

from base.base_dataset import BaseDataset
import numpy as np
import pandas as pd


class LSMDC(BaseDataset):
  """LSMDC dataset."""

  def configure_train_test_splits(self, cut_name, split_name):

    if cut_name in ["full"]:
      train_list_path = "LSMDC16_annos_training.csv"
      test_list_path = "LSMDC16_challenge_1000_publictect.csv"

      test_list_path = os.path.join(self.data_dir, test_list_path)
      df = pd.read_csv(test_list_path, delimiter="\t", header=None)
      test_vid_list = list(df[0])
      nb_test_samples = len(test_vid_list)

      if split_name in ["train", "trn", "val", "trainval"]:
        train_list_path = os.path.join(self.data_dir, train_list_path)
        df = pd.read_csv(train_list_path, delimiter="\t", header=None)
        train_vid_list = list(df[0])

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

    # There are five videos without captions in the training set, so we drop
    # them.
    movies = [
        "0024_THE_LORD_OF_THE_RINGS_THE_FELLOWSHIP_OF_THE_RING_00.31.10.217-00.31.10.706",
        "1014_2012_00.01.21.399-00.01.23.997",
        "1014_2012_00.27.58.174-00.27.59.021",
        "1018_Body_Of_Lies_00.42.15.677-00.42.18.534",
        "1037_The_Curious_Case_Of_Benjamin_Button_02.25.14.743-02.25.17.312",
    ]
    for movie in movies:
      if movie in self.vid_list:
        self.vid_list.remove(movie)

    self.split_name = split_name
    self.dataset_name = f"LSMDC_{cut_name}_{split_name}"
