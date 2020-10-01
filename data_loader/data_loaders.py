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
"""Logic for cation-video pairs dataloading."""
import logging

from data_loader.mix_dataset import MixDataset
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class ExpertDataLoader:
  """Data loading of a dataset."""

  def __init__(
      self,
      mix,
      num_workers,
      batch_size,
      raw_input_dims,
      until_epoch=float("inf"),
      pin_memory=False,
      n_pairs=1,
      training=False,
      tokenizer=None,
      loaded_data=None,
      cross_seed=0,
  ):

    self.batch_size = batch_size
    self.until_epoch = until_epoch
    self.n_pairs = n_pairs

    dataset = MixDataset(
        mix=mix,
        raw_input_dims=raw_input_dims,
        training=training,
        tokenizer=tokenizer,
        n_pairs=n_pairs,
        loaded_data=loaded_data,
        cross_seed=cross_seed,
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=dataset.collate_data,
        drop_last=training,
        shuffle=training,
        pin_memory=pin_memory,
    )
    self.dataloaders = {
        "loader": loader,
        "dataset": dataset,
    }
    logger.debug("Loading data with %d workers", num_workers)

  def __getitem__(self, key):
    return self.dataloaders[key]
