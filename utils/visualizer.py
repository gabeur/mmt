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
"""Qualitative evaluation of the results.

Code based on the implementation of "Collaborative Experts":
https://github.com/albanie/collaborative-experts
"""

import itertools
import logging
import os
import pathlib
import shutil

from . import html_utils as html
from . import util
import numpy as np

logger = logging.getLogger(__name__)


class Visualizer:
  """Visualizer class."""

  def __init__(self, exp_name, web_dirs, vis_vid_freq, num_samples=50):
    self.name = exp_name
    self.web_dirs = web_dirs
    self.vis_vid_freq = vis_vid_freq
    self.num_samples = num_samples
    logger.debug("create web directories %s...", str(self.web_dirs))
    util.mkdirs(self.web_dirs)

  def visualize_ranking(self, sims, query_masks, epoch, meta, nested_metrics,
                        modalities, subdir_name, sets, tokenizer):
    """Create html page to visualize results."""
    if not ((self.vis_vid_freq and epoch % self.vis_vid_freq == 0) or
            sets == "final_eval"):
      return
    if epoch == 0:
      return

    query_masks = query_masks.reshape(-1).astype(bool)
    nb_queries = sims.shape[0]
    nb_candidates = sims.shape[1]
    eye = np.identity(nb_candidates, dtype=float).astype(bool)
    queries_per_candidate = nb_queries / nb_candidates
    pos_mask = np.repeat(eye, queries_per_candidate, axis=0)

    # Remove the invalid captions
    pos_mask = pos_mask[query_masks]
    sims = sims[query_masks]
    meta["raw_captions"] = list(
        itertools.compress(meta["raw_captions"], query_masks))
    nb_text_weights = meta["text_weights"].shape[-1]
    meta["text_weights"] = np.reshape(meta["text_weights"],
                                      (-1, nb_text_weights))[query_masks]

    dists = -sims
    gt_dists = dists[pos_mask]

    np.random.seed(0)
    sorted_ranks = np.argsort(dists, axis=1)
    rankings = []
    vis_top_k = 5
    hide_gt = False
    size = min(dists.shape[0], self.num_samples)
    sample = np.random.choice(np.arange(dists.shape[0]),
                              size=size,
                              replace=False)
    for ii in sample:
      ranked_idx = sorted_ranks[ii][:vis_top_k]
      # gt_captions = meta["raw_captions"][ii]
      ids = meta["token_ids"][ii][:, 0].numpy()
      gt_captions = tokenizer.convert_ids_to_tokens(ids)
      gt_candidate_idx = np.where(pos_mask[ii])[0][0]

      datum = {
          "gt-sim": -gt_dists[ii],
          "gt-captions": gt_captions,
          "gt-rank": np.where(sorted_ranks[ii] == gt_candidate_idx)[0][0],
          "gt-path": meta["paths"][gt_candidate_idx],
          "text_weights": meta["text_weights"][ii],
          "ranked_idx": ranked_idx,
          "top-k-vid_weights": np.array(meta["vid_weights"])[ranked_idx],
          "top-k-sims": -dists[ii][ranked_idx],
          "top-k-paths": np.array(meta["paths"])[ranked_idx],
          "hide-gt": hide_gt,
      }
      rankings.append(datum)

    for web_dir in self.web_dirs:
      web_dir = os.path.join(web_dir, subdir_name)
      if pathlib.Path(web_dir).exists():
        try:
          shutil.rmtree(web_dir)
        except OSError as e:
          print("Error: %s : %s" % (web_dir, e.strerror))
      if not pathlib.Path(web_dir).exists():
        pathlib.Path(web_dir).mkdir(exist_ok=True, parents=True)

      self.display_current_results(
          rankings,
          epoch=epoch,
          metrics=nested_metrics["t2v_metrics"],
          modalities=modalities,
          web_dir=web_dir,
      )

  def display_current_results(self, rankings, epoch, metrics, modalities,
                              web_dir):
    """Create html page to visualize the rankings."""
    visualize_weights = True

    filepath = pathlib.Path(web_dir) / "index.html"
    if filepath.exists():
      filepath.unlink()
    pathlib.Path(web_dir).mkdir(exist_ok=True, parents=True)

    print(f"updating webpage at {web_dir}")
    title = f"Experiment name = {self.name}"
    refresh = True
    if not refresh:
      logger.debug("DISABLING WEB PAGE REFRESH")
    webpage = html.HTML(web_dir=web_dir, title=title, refresh=refresh)

    msg = f"epoch [{epoch}] - {self.name}"
    webpage.add_header(msg)
    msg = (f"R1: {metrics['R1']:.1f}, "
           f"R5: {metrics['R5']:.1f}, "
           f"R10: {metrics['R10']:.1f}, "
           f"MedR: {metrics['MedR']}")
    webpage.add_header(msg)
    logger.debug("Top %d retrieved videos at epoch: %d", len(rankings[0]),
                 epoch)

    for line_nb, ranking in enumerate(rankings):
      vids, txts, links = [], [], []
      gt_vid_path = str(ranking["gt-path"])
      gt_captions = " ".join(ranking["gt-captions"])
      gt_captions.replace(" ##", "")

      if ranking["hide-gt"]:
        txts.append(gt_captions)
        links.append("hidden")
        vids.append("hidden")
      else:
        txt = (f"<b>{line_nb + 1}<br>{gt_captions}<br><b>Rank: "
               f"{ranking['gt-rank'] + 1}, Sim: {ranking['gt-sim']:.3f} "
               f"[{ranking['gt-path'].stem}]")
        if visualize_weights:
          txt = txt + "<br><b>text weights:"
          for mod_idx, text_weight in enumerate(ranking["text_weights"]):
            mod_name = modalities[mod_idx]
            txt = txt + f"<br><b>{mod_name}: {text_weight:.2f}"
        txts.append(txt)
        links.append(gt_vid_path)
        vids.append(gt_vid_path)

      for idx, (path, sim, vid_weights) in enumerate(
          zip(ranking["top-k-paths"], ranking["top-k-sims"],
              ranking["top-k-vid_weights"])):
        if ranking["hide-gt"]:
          txt = f"choice: {idx}"
        else:
          txt = f"<b>Rank: {idx + 1}, Sim: {sim:.3f}, [{path.stem}]"

        if visualize_weights:
          txt = txt + "<br><b>video weights:"
          for mod_idx, vid_weight in enumerate(vid_weights):
            mod_name = modalities[mod_idx]
            txt = txt + f"<br><b>{mod_name}: {vid_weight:.2f}"

        txts.append(txt)
        vid_path = str(path)
        vids.append(vid_path)
        links.append(vid_path)
      webpage.add_videos(vids, txts, links, width=200)
    logger.debug("added %d videos", len(vids))
    webpage.save()
