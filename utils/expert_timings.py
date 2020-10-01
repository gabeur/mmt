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
#
# Dict used to calculate at which moment of the video was each feature extracted
expert_timings = {
    'rgb': {
        'feat_width': 0.2,
    },
    'face': {
        'feat_width': None,
    },
    'scene': {
        'feat_width': 1.0,
    },
    'speech': {
        'feat_width': None,
    },
    'ocr': {
        'feat_width': None,
    },
    's3d': {
        'feat_width': 1.0,
    },
    'vggish': {
        'feat_width': 1.0,
    },
    'audio_c': {
        'feat_width': None,
    },
    'face_c': {
        'feat_width': None,
    },
    'ocr_c': {
        'feat_width': None,
    },
    'speech_c': {
        'feat_width': None,
    },
}
