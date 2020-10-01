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
"""Utilities for updating the log where evaluation steps are recorded."""
import time


def update_perf_log(epoch_perf, perf_log_path):
  now = time.strftime('%c')
  line = 't: {}, '.format(now)
  for key in epoch_perf:
    line += '{}: {}, '.format(key, epoch_perf[key])

  line += '\n'

  with open(perf_log_path, 'a') as file:
    file.write(line)
