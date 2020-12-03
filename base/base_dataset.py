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
# pylint: disable=g-explicit-length-test
"""Logic for reading and formatting the caption-video pairs."""
import abc
import logging
import os
import pathlib
import random
import re

import h5py

import ipdb
import numpy as np
from torch.utils.data import Dataset
import utils.expert_timings as expert_timings
import utils.stop_words as stop_words
from utils.util import memcache

logger = logging.getLogger(__name__)


def is_end_of_sentence(word):
  if word[-1] in [".", "?", "!"]:
    return True
  else:
    return False


def is_stop_word(word):
  word_pure = get_clean_word(word)
  if word_pure in stop_words.ENGLISH_STOP_WORDS:
    return True
  if not word_pure.isalnum():
    return True
  for word_piece in word_pure.split("\'"):
    if word_piece in stop_words.ENGLISH_STOP_WORDS:
      return True
  return False


def get_clean_word(word):
  # Remove invalid characters from word
  word_pure = word
  invalid_char = [".", ",", "?", "!"]
  for char in invalid_char:
    word_pure = word_pure.replace(char, "")
  return word_pure.lower()


def crop_or_pad_to_len(token_ids, max_text_words):
  token_ids_tensor = np.zeros((max_text_words, 2))
  keep = min(len(token_ids), max_text_words)
  token_ids_tensor[:keep, 0] = token_ids[:keep]
  token_ids_tensor[:keep, 1] = 1
  return token_ids_tensor


def choose_or_pad_to_len(features,
                         features_t,
                         max_tokens,
                         training,
                         shuffle=False,
                         seed=0):
  """Outputs a fixed length sequence of features from a variable length input.

  Performs a selection if there are too many input features.
  Pads the sequence if there are too few features.

  Args:
    features: Input features.
    features_t: Input features timestamps.
    max_tokens: Length of the output sequence.
    training: If True, the features will be deterministically sampled.
    shuffle: If True, the features are shuffled.
    seed: Seed used for the random shuffling.

  Returns:
    Fixed length sequence of features.
  """
  feature_dim = features.shape[-1]
  tensor = np.zeros((max_tokens, feature_dim))
  tensor_t = np.ones((max_tokens))
  tensor_ind = np.zeros((max_tokens))
  keep = min(len(features), max_tokens)
  if training:
    # If training, we randomly pick features
    pick = np.random.choice(len(features), size=keep, replace=False)
  else:
    # If not training, the choice of features is deterministic
    rng = np.random.RandomState(0)
    pick = rng.choice(len(features), size=keep, replace=False)
  pick = np.sort(pick)
  tensor[:keep, :] = features[pick]
  if shuffle and training:
    # Shuffle temporal encoding so that the model cannot use temporal
    # information.
    rng = np.random.RandomState(seed)
    tensor_t[:keep] = rng.shuffle(features_t[pick])
  else:
    tensor_t[:keep] = features_t[pick]
  tensor_ind[:keep] = 1
  return tensor, tensor_t, tensor_ind


def remove_caption_stop_words(cap, cap_t):
  """Removes the stop words from a caption."""
  res = []
  res_t = []
  for i, word in enumerate(cap):
    word_t = cap_t[i]
    if not is_stop_word(word):
      res.append(get_clean_word(word))
      res_t.append(word_t)
  if len(res) < 1:
    res.append(".")
    res_t.append(np.array([0., 0.]))
  return res, res_t


class BaseDataset(Dataset):
  """Base class for a caption-video pairs dataset."""

  @abc.abstractmethod
  def configure_train_test_splits(self, cut_name, split_name):
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
               data_dir,
               raw_input_dims,
               cut_name,
               split_name,
               max_text_words=30,
               max_expert_tokens=8,
               clip_duration=float("Inf"),
               caption_length=float("Inf"),
               captions_per_video=1,
               restrict_train_captions=0,
               training=False,
               split_size=1.0,
               load_in_ram=False,
               remove_stop_words=False,
               n_pairs=1,
               tokenizer=None,
               shuffle_feats_t=False,
               loaded_data=None,
               query_shuffling="indiv",
               cross_seed=0,
               temporal_encoding_window=1):

    self.sanity_checks = False
    self.train = training
    self.data_dir = data_dir
    self.restrict_train_captions = restrict_train_captions
    self.max_text_words = max_text_words
    self.max_expert_tokens = max_expert_tokens
    self.root_feat = pathlib.Path(data_dir) / "symlinked-feats"
    self.experts = set(raw_input_dims.keys())
    self.rgb_shots = 1
    self.cut_name = cut_name
    self.split_name = split_name
    self.split_size = split_size
    self.load_in_ram = load_in_ram
    self.remove_stop_words = remove_stop_words
    self.n_pairs = n_pairs
    self.clip_duration = clip_duration
    self.caption_length = caption_length
    self.tokenizer = tokenizer
    self.shuffle_feats_t = shuffle_feats_t
    self.query_shuffling = query_shuffling
    self.cross_seed = cross_seed
    self.temporal_encoding_window = temporal_encoding_window

    self.data_aug = False
    self.max_ratio_rem = 0

    if self.cut_name == "c":
      # The challenge features are stored in pkl files
      self.reading_from = "pkl"
    else:
      # The ECCV20 paper features are stored in multiple h5 files
      self.reading_from = "mult_h5"

    self.cache_dir = os.path.join(os.path.dirname(data_dir), "vid_feat_files",
                                  self.reading_from)
    logger.debug("Cache_dir: %s", self.cache_dir)

    # This attribute can be overloaded by different datasets, so it must be set
    # before the `configure_train_test_splits() method call`
    self.restrict_test_captions = None

    # Use a single caption per video when forming training minibatches
    # (different captions from the same video may still be used across
    # different minibatches)
    if self.train:
      self.captions_per_video = 1
    else:
      self.captions_per_video = captions_per_video

    self.ordered_experts = list(raw_input_dims.keys())

    self.configure_train_test_splits(cut_name=cut_name, split_name=split_name)
    self.expert_timings = expert_timings.expert_timings

    # If split_size is type(int) it represents the number of samples that we
    # keep.
    # If split_size is type(float) it represents the ratio of the original
    # split size that we keep.
    original_size = len(self.vid_list)
    if split_size >= 2 and isinstance(split_size, int):
      nb_samples = split_size
    elif 0 <= split_size <= 1 and isinstance(split_size, float):
      nb_samples = int(split_size * original_size)

    self.vid_list = self.vid_list[:nb_samples]
    self.num_train = len(self.vid_list)

    # Display info about the dataset split size
    main_msg = f"Number of videos in {self.dataset_name}: {original_size}"
    if self.num_train == original_size:
      msg = ""
    else:
      msg = f" but we keep only {self.num_train} (split_size = {split_size})"
    logger.debug(main_msg + msg)

    # Log how many captions per video are kept
    logger.debug("We consider %s captions per video", self.captions_per_video)
    self.raw_input_dims = raw_input_dims

    visualisations = True
    if visualisations:
      logger.debug("Storing paths to enable visualisations ...")

      symlink_to_root = pathlib.Path.cwd() / "project_root"
      # If symlink to root can be accessed, follow that path
      # Otherwise, follow the current working directory
      # (that should be the project root)
      if symlink_to_root.exists():
        video_paths = [
            os.readlink(str(symlink_to_root)) / pathlib.Path(data_dir)
            / f"videos/{x}.mp4" for x in self.vid_list
        ]
      else:
        video_paths = [
            pathlib.Path.cwd() / pathlib.Path(data_dir) / f"videos/{x}.mp4"
            for x in self.vid_list
        ]

      self.video_paths = video_paths

    self.missing_val = 0

    if not os.path.exists(self.cache_dir) and self.reading_from != "pkl":
      logger.warning("%s does not exist", self.cache_dir)

    self.variable_sz_experts = self.experts
    self.flaky_experts = self.experts

    self.loaded_in_ram = False
    self.loaded_data = loaded_data
    data_source = self.dataset_name.split("_")[0]
    if data_source not in self.loaded_data:
      self.loaded_data[data_source] = {}
    if self.load_in_ram:
      logger.info("Loading dataset {self.dataset_name} in ram ...")
      if self.reading_from == "mult_h5":
        self.data_vid = {}
        for i, vid in enumerate(self.vid_list):
          if i % 100 == 0:
            logger.debug(i)
          self.data[vid] = self.get_sample_data(vid)
      elif self.reading_from == "pkl":
        self.data_exp = self.loaded_data[data_source]
        for expert in self.experts:
          if expert not in self.data_exp:
            self.data_exp[expert] = {}
          if expert in self.expert_paths.keys():
            for agg, path in self.expert_paths[expert].items():
              data_path = pathlib.Path(self.data_dir) / pathlib.Path(path)
              if agg not in self.data_exp[expert]:
                self.data_exp[expert][agg] = memcache(data_path)
          else:
            logger.warning("The expert %s is not available for dataset %s",
                           expert, self.dataset_name)

        if self.split_name == "test2":
          path = self.expert_paths["raw_captions_test2"]
        else:
          path = self.expert_paths["raw_captions"]
        data_path = pathlib.Path(self.data_dir) / pathlib.Path(path)
        additionnal_captions = memcache(data_path)
        if "raw_captions" not in self.data_exp:
          self.data_exp["raw_captions"] = {}
        self.data_exp["raw_captions"].update(additionnal_captions)
      self.loaded_in_ram = True

  def tokenize_caption(self, raw_caption, special_tokens=True):
    tokenize = True

    word_list = raw_caption
    if len(word_list) == 0:
      # Empty list of words.
      ipdb.set_trace()
    if tokenize:
      txt_caption = " ".join(word_list)
      # Remove whitespace at beginning and end of the sentence.
      txt_caption = txt_caption.strip()
      # Add period at the end of the sentence if not already there.
      if txt_caption[-1] not in [".", "?", "!"]:
        txt_caption += "."
      txt_caption = txt_caption.capitalize()
      tokens = self.tokenizer.tokenize(txt_caption)
      if special_tokens:
        cls = [self.tokenizer.cls_token]
        sep = [self.tokenizer.sep_token]  # [SEP] token
        tokens = cls + tokens + sep
      tokens = tokens[:self.max_text_words]
      # Make sure that the last token is
      # the [SEP] token
      if special_tokens:
        tokens[-1] = self.tokenizer.sep_token

      ids = self.tokenizer.convert_tokens_to_ids(tokens)
    else:
      ids = list(range(len(word_list)))

    if len(ids) <= 0:
      ipdb.set_trace()

    return ids



  def get_feature_timings(self, nb_feats, feat_width, stride=None, group=None):
    # Return an array containing the start time of each feature in the first
    # line and the end time of each feature in the second line.
    if feat_width is None:
      timings = np.empty((nb_feats, 2))
      timings[:] = -1
      return timings
    if group is not None:
      assert nb_feats % group == 0
      nb_feats_top = nb_feats // group
      top_timings = self.get_feature_timings(nb_feats_top,
                                             feat_width,
                                             stride,
                                             group=None)
      bot_timings = np.repeat(top_timings, group, axis=-1)
      return bot_timings
    if stride is None:
      stride = feat_width
    starts = np.linspace(0, (nb_feats - 1) * stride, num=nb_feats)
    ends = np.linspace(feat_width, (nb_feats - 1) * stride + feat_width,
                       num=nb_feats)
    res = np.stack((starts, ends), axis=-1)
    return res

  def aggregate_feats(self, feats, mode):
    assert feats.ndim == 2
    if mode == "avg":
      agg = np.mean(feats, axis=0, keepdims=True)
    elif mode == "max":
      agg = np.max(feats, axis=0, keepdims=True)
    else:
      msg = "aggregation mode {} not supported"
      raise NotImplementedError(msg.format(mode))
    return agg

  def collate_data(self, data):
    text_keys = data[0]["text_tensors"].keys()
    text_tensors = {key: [] for key in text_keys}

    vid_keys = data[0]["vid_tensors"].keys()
    vid_tensors = {
        key: {expert: [] for expert in self.experts} for key in vid_keys
    }

    l_keys = data[0]["lists"].keys()
    lists = {key: [] for key in l_keys}

    for vid in data:
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

  def get_sample_data(self, vid):
    if self.reading_from == "mult_h5":
      if self.loaded_in_ram:
        return self.data_vid[vid]
      else:
        if vid.endswith(".h5"):
          dataset_file_path = vid
        else:
          output_basename = f"{vid[0]}/{vid[1]}/{vid[2]}/{vid}.h5"
          dataset_file_path = os.path.join(self.cache_dir, output_basename)

        dataset_file = h5py.File(dataset_file_path, "r")
        with h5py.File(dataset_file_path, "r") as dataset_file:
          video_data = dataset_file
          keys_list = list(video_data.keys())
          nb_captions = len(
              [k for k in keys_list if k.startswith("raw_captions.")])
          if nb_captions == 0:
            logger.warning("No caption for %s", dataset_file_path)
          assert nb_captions > 0
          raw_captions = []
          raw_captions_t = []
          for i in range(nb_captions):
            raw_caption = video_data[f"raw_captions.{i}"].value
            raw_captions.append(raw_caption)
            if f"raw_captions_t.{i}" in video_data.keys():
              raw_caption_t = video_data[f"raw_captions_t.{i}"].value
              if raw_caption_t.shape[0] != len(raw_caption):
                raw_caption_t = raw_caption_t[:len(raw_caption)]
              raw_captions_t.append(raw_caption_t)
            else:
              nb_words = len(raw_caption)
              raw_caption_t = np.zeros((nb_words, 2))
              raw_captions_t.append(raw_caption_t)

          features = {}
          features_t = {}
          features_avgpool = {}
          features_maxpool = {}
          for expert in self.experts:
            if f"features.{expert}" in video_data.keys():
              x = video_data[f"features.{expert}"].value
              if len(x) > 0 and not np.isnan(x[0][0]):
                features[expert] = video_data[f"features.{expert}"].value

                if f"features_t.{expert}" in video_data.keys():
                  x = video_data[f"features_t.{expert}"].value
                  # if not np.isnan(x[0][0]):
                  if expert in ["s3d", "vggish"]:
                    features_t[expert] = video_data[
                        f"features_t.{expert}"].value
                    if features_t[expert].shape[0] != features[expert].shape[0]:
                      logger.warning(
                          "Incorrect number of features_t values "
                          "for %s", dataset_file_path)
                      features_t[expert] = features_t[expert][:features[expert].
                                                              shape[0]]
                  else:
                    nb_feats = features[expert].shape[0]
                    expert_timing = self.expert_timings[expert]
                    features_t[expert] = self.get_feature_timings(
                        nb_feats, **expert_timing)
                else:
                  nb_feats = features[expert].shape[0]
                  expert_timing = self.expert_timings[expert]
                  features_t[expert] = self.get_feature_timings(
                      nb_feats, **expert_timing)
                features_t[expert] = np.average(features_t[expert], axis=1)
            features_avgpool[expert] = None
            features_maxpool[expert] = None
        return (raw_captions, raw_captions_t, features, features_t,
                features_avgpool, features_maxpool)

    elif self.reading_from == "pkl":
      if self.loaded_in_ram:
        # Raw captions
        video_data = self.data_exp["raw_captions"]
        if vid in video_data.keys():
          x = video_data[vid]
          nb_captions = len(x)
          assert nb_captions > 0
          raw_captions = video_data[vid]
          raw_captions_t = []
          for i in range(nb_captions):
            raw_caption = raw_captions[i]
            nb_words = len(raw_caption)
            raw_caption_t = np.zeros((nb_words, 2))
            raw_captions_t.append(raw_caption_t)

        # Video features
        features = {}
        features_t = {}
        for expert in self.experts:
          if "fixed_seg" not in self.data_exp[expert].keys():
            continue
          video_data = self.data_exp[expert]["fixed_seg"]
          if vid in video_data.keys():
            x = video_data[vid]
            if not isinstance(x, float) and x and not np.isnan(x[0][0]):
              features[expert] = x

              # Timings
              nb_feats = features[expert].shape[0]
              if expert in self.expert_timings.keys():
                expert_timing = self.expert_timings[expert]
              else:
                expert_timing = {"feat_width": 1.0}
              features_t[expert] = self.get_feature_timings(
                  nb_feats, **expert_timing)
              features_t[expert] = np.average(features_t[expert], axis=1)

        features_avgpool = {}
        agg = "avg"
        for expert in self.experts:
          if agg not in self.data_exp[expert].keys():
            features_avgpool[expert] = None
            continue
          video_data = self.data_exp[expert][agg]
          if vid in video_data.keys():
            x = video_data[vid]
            if len(x) > 0 and not np.isnan(x[0][0]):
              features_avgpool[expert] = x

        features_maxpool = {}
        agg = "max"
        for expert in self.experts:
          if agg not in self.data_exp[expert].keys():
            features_maxpool[expert] = None
            continue
          video_data = self.data_exp[expert][agg]
          if vid in video_data.keys():
            x = video_data[vid]
            if len(x) > 0 and not np.isnan(x[0][0]):
              features_maxpool[expert] = x

        return (raw_captions, raw_captions_t, features, features_t,
                features_avgpool, features_maxpool)

  def __len__(self):
    if self.train:
      # If it is a training dataset, we let the trainer decide when the epoch
      # is completed.
      return max(self.num_train, 1e6)
    else:
      return self.num_train

  def __getitem__(self, idx):

    idx = idx % self.num_train

    vid = self.vid_list[idx]
    path = self.video_paths[idx]

    (captions, captions_t, features, features_t, features_avgpool_provided,
     features_maxpool_provided) = self.get_sample_data(vid)

    if self.restrict_test_captions and vid in self.restrict_test_captions.keys(
    ):
      keep_sent_idx = self.restrict_test_captions[vid]
      # text = [text[keep_sent_idx]]
      captions = [captions[keep_sent_idx]]
      captions_t = [captions_t[keep_sent_idx]]

    raw_captions = []
    raw_captions_t = []

    captions_picked = min(len(captions), self.captions_per_video)
    for cap_nb in range(captions_picked):

      # For the evaluation
      # (no shuffling, only ok if self.captions_per_video != 1)
      if self.query_shuffling == "indiv":
        # Not concatenating the captions
        raw_captions.append(captions[cap_nb])
        raw_captions_t.append(captions_t[cap_nb])

      if self.query_shuffling == "cat":
        # Concatenating the captions and keeping the original order
        raw_captions.append(np.concatenate(captions))
        raw_captions_t.append(np.concatenate(captions_t))

      if self.query_shuffling == "shuf":
        # Shuffling then concatenating the captions
        c = list(zip(captions, captions_t))
        random.shuffle(c)
        captions, captions_t = zip(*c)
        raw_captions.append(np.concatenate(captions))
        raw_captions_t.append(np.concatenate(captions_t))

      z = re.match(r"shufk(\d*)", self.query_shuffling)
      if z:
        # Shuffling then concatenating the captions and keeping the first few
        nb_keep = min(int(z.groups()[0]), len(captions))
        c = list(zip(captions, captions_t))
        random.shuffle(c)
        captions, captions_t = zip(*c)
        keep_captions = captions[:nb_keep]
        keep_captions_t = captions_t[:nb_keep]
        raw_captions.append(np.concatenate(keep_captions))
        raw_captions_t.append(np.concatenate(keep_captions_t))

    raw_captions = np.array(raw_captions, dtype=object)
    raw_captions_t = np.array(raw_captions_t)

    paths = []
    sources = []
    raw_captions_list = []
    raw_captions_t_list = []
    token_ids_list = []
    query_masks_list = []
    features_dic = {}
    features_t_dic = {}
    features_ind_dic = {}
    features_avgpool_dic = {}
    features_maxpool_dic = {}
    for expert in self.experts:
      features_dic[expert] = []
      features_t_dic[expert] = []
      features_ind_dic[expert] = []
      features_avgpool_dic[expert] = []
      features_maxpool_dic[expert] = []

    split_sentences_list = []
    for cap_idx in range(self.captions_per_video):
      if cap_idx < len(raw_captions):
        cap = np.array([
            el if isinstance(el, str) else el.decode("UTF-8")
            for el in raw_captions[cap_idx]
        ])
        cap_t = np.array(raw_captions_t[cap_idx])

        # HowTo100M have no video features extracted beyond 500s, so ignore text
        # after that.
        keep_ids = cap_t[:, 0] < 500
        cap = np.expand_dims(cap[keep_ids], axis=-1)
        cap_t = np.expand_dims(cap_t[keep_ids], axis=-1)
        if len(cap) < 1:
          # The cap length can be 0 when there are no words pronounced in the
          # first 500s.
          cap = np.array([["."]])
          cap_t = np.array([[[0, 0]]])
      else:
        # Requested more captions than available, zero padding
        cap = np.array([["0"]])
        cap_t = np.array([[[0, 0]]])

      split_sentences_list.append((cap, cap_t))

    captions_query_masks = np.zeros((self.captions_per_video))
    captions_query_masks[:len(raw_captions)] = 1

    # n_pairs number of clips to sample from each video
    for _ in range(self.n_pairs):
      token_ids = []
      raw_captions_ = []
      raw_captions_t_ = []
      for cap_idx in range(self.captions_per_video):
        if self.train:
          rng = np.random
        else:
          # Same data selection across epochs
          rng = np.random.RandomState(idx)

        # Number of sentences to pick per caption
        if isinstance(self.caption_length, list):
          min_picked_sentences = self.caption_length[0]
          max_picked_sentences = self.caption_length[1]
        else:
          min_picked_sentences = self.caption_length
          max_picked_sentences = self.caption_length

        if min_picked_sentences == float("Inf"):
          nb_sentences = float("Inf")
        else:
          nb_sentences = rng.randint(min_picked_sentences,
                                     max_picked_sentences + 1)

        # Duration in seconds of the video crop to consider
        if isinstance(self.clip_duration, list):
          clip_duration_min = self.clip_duration[0]
          clip_duration_max = self.clip_duration[1]
        else:
          clip_duration_min = self.clip_duration
          clip_duration_max = self.clip_duration

        if clip_duration_max == float("Inf"):
          clip_length = float("Inf")
        else:
          clip_length = rng.uniform(clip_duration_min, clip_duration_max)

        sentences, sentences_t = split_sentences_list[cap_idx]

        # The number of sentences that we will keep is the min between the
        # number of sentences requested and the number of available sentences.
        nb_sentences = min(nb_sentences, len(sentences))
        choice = rng.randint(len(sentences) + 1 - nb_sentences)

        selected_sentences = sentences[choice:choice + nb_sentences]
        selected_sentences_t = sentences_t[choice:choice + nb_sentences]
        if len(selected_sentences) < 1:
          ipdb.set_trace()
          print(f"Did not select any sentence for vid {vid} of idx {idx}")
          print(sentences)

        selected_sentences = np.concatenate(selected_sentences)
        selected_sentences_t = np.concatenate(selected_sentences_t)
        if self.remove_stop_words:
          selected_sentences, selected_sentences_t = remove_caption_stop_words(
              selected_sentences, selected_sentences_t)

        # np.array(["aa", "bb"])
        selected_sentences = selected_sentences[:self.max_text_words]
        selected_sentences_t = selected_sentences_t[:self.max_text_words]

        raw_captions_.append(
            selected_sentences)  # [cap_per_vid * np.array(["aa", "bb"])]
        raw_captions_t_.append(selected_sentences_t)

        caption_token_ids = self.tokenize_caption(selected_sentences,
                                                  special_tokens=True)
        caption_token_ids = crop_or_pad_to_len(caption_token_ids,
                                               self.max_text_words)
        token_ids.append(caption_token_ids)

      # [n_pairs * np.array([cap_per_vid * np.array(["aa", "bb"])])]
      raw_captions_list.append(np.array(raw_captions_, dtype=object))
      raw_captions_t_list.append(np.array(raw_captions_t_, dtype=object))

      captions_token_ids = np.stack(
          token_ids, axis=0)  # (captions_per_video, max_text_words, 2)
      token_ids_list.append(captions_token_ids)
      query_masks_list.append(captions_query_masks)

      if clip_length == float("inf"):
        feat_start = 0
        feat_end = float("inf")
      else:
        # Calculate the mean time of when the kept sentences are pronounced
        sentences_start = np.min(selected_sentences_t)
        sentences_end = np.max(selected_sentences_t)
        c_time = np.mean((sentences_start, sentences_end))

        # We sample a window centered on c_time to sample video features from
        feat_start = c_time - clip_length / 2
        feat_end = feat_start + clip_length

      for expert in self.experts:
        raw_dim = self.raw_input_dims[expert]
        if expert in features.keys():
          if clip_length == float("inf"):
            features_sel = features[expert]
            # For temporal encoding, we want the temporal features to start at
            # 2s.
            features_t_sel = (features_t[expert]
                              - feat_start) / self.temporal_encoding_window
            features_t_sel = features_t_sel + 2
          else:
            indices_to_keep = np.logical_and(feat_start <= features_t[expert],
                                             features_t[expert] <= feat_end)
            nb_kept = indices_to_keep.sum()
            if nb_kept > 0:
              features_sel = features[expert][indices_to_keep]
              # For temporal encoding, we want the temporal features to start at
              # 2s.
              features_t_sel = (features_t[expert][indices_to_keep]
                                - feat_start) / self.temporal_encoding_window
              features_t_sel = features_t_sel + 2
            else:
              # No features kept for the expert
              features_sel = None
        else:
          # No features available for the expert
          features_sel = None

        if features_sel is None:
          # No features available for the expert
          features_sel = np.zeros((self.max_expert_tokens, raw_dim))
          features_t_sel = np.ones((self.max_expert_tokens))
          features_avgpool = np.zeros((1, raw_dim))
          features_maxpool = np.zeros((1, raw_dim))
          features_ind_sel = np.zeros((self.max_expert_tokens))
        else:
          # Aggregated video features before choosing a subset
          assert features_sel.ndim == 2
          assert features_sel.shape[1] == self.raw_input_dims[expert]

          features_avgpool = self.aggregate_feats(
              feats=features_sel,
              mode="avg",
          )
          assert features_avgpool.ndim == 2

          features_maxpool = self.aggregate_feats(
              feats=features_sel,
              mode="max",
          )
          assert features_maxpool.ndim == 2

          # Choose a subset of the video features to fit in memory
          features_sel, features_t_sel, features_ind_sel = choose_or_pad_to_len(
              features_sel,
              features_t_sel,
              self.max_expert_tokens,
              self.train,
              shuffle=self.shuffle_feats_t,
              seed=idx)

        if features_avgpool_provided[expert] is not None:
          features_avgpool = features_avgpool_provided[expert]
          assert features_avgpool.ndim == 2
        if features_maxpool_provided[expert] is not None:
          features_maxpool = features_maxpool_provided[expert]
          assert features_maxpool.ndim == 2

        features_dic[expert].append(features_sel)
        features_t_dic[expert].append(features_t_sel)
        features_avgpool_dic[expert].append(features_avgpool)
        features_maxpool_dic[expert].append(features_maxpool)
        features_ind_dic[expert].append(features_ind_sel)

      paths.append(path)
      sources.append(self.dataset_name)

    # Batch together all the pairs extracted from the video
    features_ind = {}
    features_avgpool = {}
    features_maxpool = {}
    token_ids = np.stack(
        token_ids_list,
        axis=0)  # (n_pairs, captions_per_video, max_text_words, 2)
    query_masks = np.stack(query_masks_list,
                           axis=0)  # (n_pairs, captions_per_video)
    for expert in self.experts:
      features_list = features_dic[
          expert]  # [n_pairs x (max_expert_tokens, feat_dim)]
      features[expert] = np.stack(
          features_list, axis=0)  # (n_pairs, max_expert_tokens, feat_dim)
      features_t_list = features_t_dic[
          expert]  # [n_pairs x (max_expert_tokens)]
      features_t[expert] = np.stack(features_t_list,
                                    axis=0)  # (n_pairs, max_expert_tokens)
      features_ind_list = features_ind_dic[
          expert]  # [n_pairs x (max_expert_tokens)]
      features_ind[expert] = np.stack(features_ind_list,
                                      axis=0)  # (n_pairs, max_expert_tokens)
      features_avgpool[expert] = np.concatenate(features_avgpool_dic[expert],
                                                axis=0)  # (n_pairs, feat_dim)
      features_maxpool[expert] = np.concatenate(features_maxpool_dic[expert],
                                                axis=0)  # (n_pairs, feat_dim)

    text_tensors = {"token_ids": token_ids, "query_masks": query_masks}
    vid_tensors = {
        "features": features,
        "features_t": features_t,
        "features_ind": features_ind,
        "features_avgpool": features_avgpool,
        "features_maxpool": features_maxpool
    }

    lists = {
        "raw_captions": raw_captions_list,
        "raw_captions_t": raw_captions_t_list,
        "paths": paths,
        "sources": sources
    }

    return {
        "text_tensors": text_tensors,
        "vid_tensors": vid_tensors,
        "lists": lists
    }
