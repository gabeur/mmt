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
"""Utilities to compute timing statistics."""


class AverageMeter(object):
  """Computes and stores the average and current value."""

  def __init__(self):

    self.dic = {}

    self.reset()

  def reset(self):
    for key in self.dic:
      for metric in self.dic[key]:
        self.dic[key][metric] = 0

  def update(self, key, val, n=1):
    self.dic.setdefault(key, {'val': 0, 'sum': 0, 'count': 0, 'avg': 0})

    self.dic[key]['val'] = val
    self.dic[key]['sum'] += val * n
    self.dic[key]['count'] += n
    self.dic[key]['avg'] = self.dic[key]['sum'] / self.dic[key]['count']
