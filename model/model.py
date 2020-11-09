# Copyright 2020 Valentin Gabeur
# Copyright 2020 Samuel Albanie, Yang Liu and Arsha Nagrani
# Copyright 2018 Antoine Miech All Rights Reserved.
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
"""Cross-modal Architecture models.

Code based on the implementation of "Collaborative Experts":
https://github.com/albanie/collaborative-experts

Code based on the implementation of "Mixture of Embedding Experts":
https://github.com/antoine77340/Mixture-of-Embedding-Experts
"""

import collections
import itertools
import logging
import re
import types

from base import BaseModel
from model.bert import BertModel
from model.lstm import LSTMModel
from model.net_vlad import NetVLAD
from model.txt_embeddings import TxtEmbeddings
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_bert import BertModel as TxtBertModel
from utils.util import get_len_sequences

logger = logging.getLogger(__name__)


class CENet(BaseModel):
  """Whole cross-modal architecture."""

  def __init__(self,
               l2renorm,
               expert_dims,
               tokenizer,
               keep_missing_modalities,
               test_caption_mode,
               freeze_weights=False,
               mimic_ce_dims=False,
               concat_experts=False,
               concat_mix_experts=False,
               use_experts='origfeat',
               txt_inp=None,
               txt_agg=None,
               txt_pro=None,
               txt_wgh=None,
               vid_inp=None,
               vid_cont=None,
               vid_wgh=None,
               pos_enc=None,
               out_tok=None,
               use_mask='nomask',
               same_dim=512,
               vid_bert_params=None,
               txt_bert_params=None,
               agg_dims=None,
               normalize_experts=True):
    super().__init__()

    self.sanity_checks = False
    modalities = list(expert_dims.keys())
    self.expert_dims = expert_dims
    self.modalities = modalities
    logger.debug(self.modalities)
    self.mimic_ce_dims = mimic_ce_dims
    self.concat_experts = concat_experts
    self.concat_mix_experts = concat_mix_experts
    self.test_caption_mode = test_caption_mode
    self.freeze_weights = freeze_weights
    self.use_experts = use_experts
    self.use_mask = use_mask
    self.keep_missing_modalities = keep_missing_modalities
    self.l2renorm = l2renorm
    self.same_dim = same_dim
    self.txt_inp = txt_inp
    self.txt_agg = txt_agg
    self.txt_pro = txt_pro
    self.txt_wgh = txt_wgh
    self.vid_inp = vid_inp
    self.vid_cont = vid_cont
    self.vid_wgh = vid_wgh
    self.pos_enc = pos_enc
    self.out_tok = out_tok
    self.vid_bert_params = vid_bert_params
    self.normalize_experts = normalize_experts

    self.video_dim_reduce = nn.ModuleDict()
    for mod in self.modalities:
      in_dim = expert_dims[mod]['dim']
      if self.vid_inp in ['agg', 'both', 'all', 'temp']:
        self.video_dim_reduce[mod] = ReduceDim(in_dim, same_dim)

    if self.vid_cont == 'coll':
      self.g_reason_1 = nn.Linear(same_dim * 2, same_dim)
      dout_prob = vid_bert_params['hidden_dropout_prob']
      self.coll_g_dropout = nn.Dropout(dout_prob)
      self.g_reason_2 = nn.Linear(same_dim, same_dim)

      self.f_reason_1 = nn.Linear(same_dim, same_dim)
      self.coll_f_dropout = nn.Dropout(dout_prob)
      self.f_reason_2 = nn.Linear(same_dim, same_dim)
      self.f_reason_3 = nn.Linear(same_dim, same_dim)
      self.batch_norm_g1 = nn.BatchNorm1d(same_dim)
      self.batch_norm_g2 = nn.BatchNorm1d(same_dim)

      self.batch_norm_f1 = nn.BatchNorm1d(same_dim)
      self.batch_norm_f2 = nn.BatchNorm1d(same_dim)

      self.video_GU = nn.ModuleDict()
      for mod in self.modalities:
        self.video_GU[mod] = GatedEmbeddingUnitReasoning(same_dim)

    # If Bert architecture is employed for video
    elif self.vid_cont == 'bert':
      vid_bert_config = types.SimpleNamespace(**vid_bert_params)
      self.vid_bert = BertModel(vid_bert_config)

    elif self.vid_cont == 'none':
      pass

    if self.txt_agg[:4] in ['bert']:
      z = re.match(r'bert([a-z]{3})(\d*)(\D*)', txt_agg)
      assert z
      state = z.groups()[0]
      freeze_until = z.groups()[1]

      # Post aggregation: Use [CLS] token ("cls") or aggregate all tokens
      # (mxp, mnp)
      if z.groups()[2] and z.groups()[2] != 'cls':
        self.post_agg = z.groups()[2]
      else:
        self.post_agg = 'cls'

      if state in ['ftn', 'frz']:
        # State is finetune or frozen, we use a pretrained bert model
        txt_bert_config = 'bert-base-cased'

        # Overwrite config
        if txt_bert_params is None:
          dout_prob = vid_bert_params['hidden_dropout_prob']
          txt_bert_params = {
              'hidden_dropout_prob': dout_prob,
              'attention_probs_dropout_prob': dout_prob,
          }
        self.txt_bert = TxtBertModel.from_pretrained(txt_bert_config,
                                                     **txt_bert_params)

        if state == 'frz':
          if freeze_until:
            # Freeze only certain layers
            freeze_until = int(freeze_until)
            logger.debug('Freezing text bert until layer %d excluded',
                         freeze_until)
            # Freeze net until given layer
            for name, param in self.txt_bert.named_parameters():
              module = name.split('.')[0]
              if name.split('.')[2].isdigit():
                layer_nb = int(name.split('.')[2])
              else:
                continue
              if module == 'encoder' and layer_nb in range(freeze_until):
                param.requires_grad = False
                logger.debug(name)
          else:
            # Freeze the whole model
            for name, param in self.txt_bert.named_parameters():
              module = name.split('.')[0]
              if module == 'encoder':
                param.requires_grad = False
        else:
          assert not freeze_until

      if self.txt_inp == 'bertfrz':
        # Freeze model
        for param in self.txt_bert.embeddings.parameters():
          param.requires_grad = False
      elif self.txt_inp not in ['bertftn']:
        logger.error('Wrong parameter for the text encoder')
      text_dim = self.txt_bert.config.hidden_size
    elif self.txt_agg in ['vlad', 'mxp', 'mnp', 'lstm']:
      # Need to get text embeddings
      if self.txt_inp == 'bertfrz':
        ckpt = 'data/word_embeddings/bert/ckpt_from_huggingface.pth'
        self.word_embeddings = TxtEmbeddings(ckpt=ckpt, freeze=True)
      elif self.txt_inp == 'bertftn':
        ckpt = 'data/word_embeddings/bert/ckpt_from_huggingface.pth'
        self.word_embeddings = TxtEmbeddings(ckpt=ckpt)
      elif self.txt_inp == 'bertscr':
        vocab_size = 28996
        emb_dim = 768
        self.word_embeddings = TxtEmbeddings(vocab_size, emb_dim)
      else:
        self.word_embeddings = tokenizer.we_model
      emb_dim = self.word_embeddings.text_dim

      if self.txt_agg == 'vlad':
        self.text_pooling = NetVLAD(
            feature_size=emb_dim,
            cluster_size=28,
        )
        text_dim = self.text_pooling.out_dim
      elif self.txt_agg == 'mxp':
        text_dim = emb_dim
      elif self.txt_agg == 'lstm':
        input_dim = self.word_embeddings.text_dim
        hidden_dim = 512
        layer_dim = 1
        output_dim = hidden_dim
        self.text_pooling = LSTMModel(input_dim, hidden_dim, layer_dim,
                                      output_dim)
        text_dim = output_dim

    self.text_GU = nn.ModuleDict()
    for mod in self.modalities:
      if self.txt_pro == 'gbn':
        self.text_GU[mod] = GatedEmbeddingUnit(text_dim,
                                               same_dim,
                                               use_bn=True,
                                               normalize=self.normalize_experts)
      elif self.txt_pro == 'gem':
        self.text_GU[mod] = GatedEmbeddingUnit(text_dim,
                                               same_dim,
                                               use_bn=False,
                                               normalize=self.normalize_experts)
      elif self.txt_pro == 'lin':
        self.text_GU[mod] = ReduceDim(text_dim, same_dim)

    # Weightening of each modality similarity
    if self.txt_wgh == 'emb':
      self.moe_fc_txt = nn.ModuleDict()
      dout_prob = txt_bert_params['hidden_dropout_prob']
      self.moe_txt_dropout = nn.Dropout(dout_prob)
      for mod in self.modalities:
        self.moe_fc_txt[mod] = nn.Linear(text_dim, 1)
    if self.vid_wgh == 'emb':
      self.moe_fc_vid = nn.ModuleDict()
      dout_prob = vid_bert_params['hidden_dropout_prob']
      self.moe_vid_dropout = nn.Dropout(dout_prob)
      for mod in self.modalities:
        self.moe_fc_vid[mod] = nn.Linear(self.same_dim, 1)

    self.debug_dataloader = False
    if self.debug_dataloader:
      self.tokenizer = tokenizer

  def compute_weights_from_emb(self, embd):
    # Compute the modality weights given an embedding

    # vid emb
    if len(embd.size()) == 2:
      embd = self.moe_vid_dropout(embd)
      moe_weights = th.cat(
          [self.moe_fc_vid[mod](embd) for mod in self.modalities], dim=-1)
      moe_weights = F.softmax(moe_weights, dim=1)

    # text emb
    elif len(embd.size()) == 3:
      embd = self.moe_txt_dropout(embd)
      b, k, d = embd.size()
      m = len(self.modalities)
      embd = embd.view(b * k, d)
      moe_weights = th.cat(
          [self.moe_fc_txt[mod](embd) for mod in self.modalities], dim=-1)
      moe_weights = F.softmax(moe_weights, dim=1)
      moe_weights = moe_weights.view(b, k, m)

    return moe_weights

  def compute_weights_from_norm(self, embds):
    # Compute the modality weights according to their norm

    device = embds[self.modalities[0]].device
    # vid emb
    if len(embds[self.modalities[0]].size()) == 2:
      b, d = embds[self.modalities[0]].size()

    # text emb
    elif len(embds[self.modalities[0]].size()) == 3:
      b, k, d = embds[self.modalities[0]].size()
      for idx, mod in self.modalities:
        embds[mod] = embds[mod].view(b * k, d)
      b = b * k

    m = len(self.modalities)
    norm_embd = th.zeros(b, m).to(device)
    for idx, mod in enumerate(self.modalities):
      norm_embd[:, idx] = th.norm(embds[mod], p=2, dim=1)

    sum_norm = th.sum(norm_embd, dim=1)  # b
    sum_norm = sum_norm.unsqueeze(1)  # b x 1

    weights = th.div(norm_embd, sum_norm)

    return weights

  def forward(self,
              token_ids,
              features,
              features_t,
              features_ind,
              features_avgpool,
              features_maxpool,
              query_masks,
              out='conf',
              device=None,
              debug=None):

    self.device = device
    experts_feats = features
    experts_feats_t = features_t
    experts_feats_ind = features_ind
    ind = {}
    for mod in self.modalities:
      ind[mod] = th.max(experts_feats_ind[mod], 1)[0]
    pooled_experts = {}

    for _, mod in enumerate(self.modalities):
      pooled_experts[f'{mod}_avgpool'] = features_avgpool[mod]
      pooled_experts[f'{mod}_maxpool'] = features_maxpool[mod]

    # Notation: B = batch size, M = number of modalities

    # Output experts
    experts = collections.OrderedDict()

    # Pass text embeddings through gated units
    text_embd = {}

    # Unroll repeated captions into present minibatch
    b, captions_per_video, max_text_words, feat_dim = token_ids.size()
    m = len(self.modalities)

    if self.txt_agg[:4] in ['bert']:
      token_ids = token_ids.view(b * captions_per_video, max_text_words,
                                 feat_dim)

      input_ids_list = []
      token_type_ids_list = []  # Modality id
      position_ids_list = []  # Position
      attention_mask_list = []  # Valid token or not

      ids_size = (b * captions_per_video,)

      for pos_id in range(max_text_words):
        input_ids_list.append(token_ids[:, pos_id, 0].to(dtype=th.long))
        token_type_ids_list.append(th.full(ids_size, 0, dtype=th.long))
        position_ids_list.append(th.full(ids_size, pos_id, dtype=th.long))
        attention_mask_list.append(token_ids[:, pos_id, 1].to(dtype=th.long))

      input_ids = th.stack(input_ids_list, dim=1).to(device)
      token_type_ids = th.stack(token_type_ids_list, dim=1).to(device)
      position_ids = th.stack(position_ids_list, dim=1).to(device)
      attention_mask = th.stack(attention_mask_list, dim=1).to(device)

      txt_bert_output = self.txt_bert(input_ids,
                                      attention_mask=attention_mask,
                                      token_type_ids=token_type_ids,
                                      position_ids=position_ids,
                                      head_mask=None)
      last_layer = txt_bert_output[0]

      if self.post_agg == 'cls':
        text = last_layer[:, 0]

      elif self.post_agg == 'mxp':
        embeddings = last_layer[:, 1:]
        text, _ = th.max(embeddings, 1)

      elif self.post_agg == 'mnp':
        embeddings = last_layer[:, 1:]
        text = th.mean(embeddings, 1)

    elif self.txt_agg in ['vlad', 'mxp', 'mnp', 'lstm']:
      # Need to get text embeddings
      token_ids = token_ids.view(b * captions_per_video, max_text_words,
                                 feat_dim)

      input_ids = token_ids[:, :, 0].to(dtype=th.long)
      attention_mask = token_ids[:, :, 1].to(dtype=th.long)

      word_embs = self.word_embeddings(input_ids)

      if self.txt_agg == 'mxp':
        # max pooling
        word_embs[attention_mask == 0] = -float('Inf')
        text = th.max(word_embs, dim=1)[0]

      elif self.txt_agg == 'vlad':
        text = self.text_pooling(word_embs)

      elif self.txt_agg == 'lstm':
        x_lengths = get_len_sequences(attention_mask)
        text = self.text_pooling(word_embs, x_lengths)

    # From the text representation, compute as many embeddings as there are
    # modalities
    for mod in self.modalities:
      layer = self.text_GU[mod]
      text_ = layer(text)
      text_ = text_.view(b, captions_per_video, -1)
      text_embd[mod] = text_
    text = text.view(b, captions_per_video, -1)

    if self.vid_inp in ['agg', 'both', 'all']:
      agg_experts = collections.OrderedDict()
      mnp_experts = collections.OrderedDict()
      maxp_experts = collections.OrderedDict()

      # Embed all features to a common dimension
      for mod in self.modalities:
        layer = self.video_dim_reduce[mod]
        mnp_experts[mod] = layer(pooled_experts[f'{mod}_avgpool'])
        maxp_experts[mod] = layer(pooled_experts[f'{mod}_maxpool'])

      for mod in self.modalities:
        agg_experts[mod] = maxp_experts[mod]

    if self.vid_inp in ['both', 'temp', 'all']:
      for mod in self.modalities:
        layer = self.video_dim_reduce[mod]
        experts_feats[mod] = layer(experts_feats[mod])

    if self.vid_cont in ['none', 'coll']:
      for _, modality in enumerate(self.modalities):
        experts[modality] = agg_experts[modality]

    # If we use collaborative gating to compute a mask (called T in the
    # Collaborative experts paper)
    if self.vid_cont == 'coll':
      masks = {}
      all_combinations = list(itertools.permutations(agg_experts, 2))
      assert len(self.modalities) > 1, 'use_ce requires multiple modalities'

      for _, modality in enumerate(self.modalities):

        mask_num = 0
        curr_mask = 0
        temp_dict = {}
        avai_dict = {}

        for modality_pair in all_combinations:
          mod0, mod1 = modality_pair
          if mod0 == modality:
            new_key = '_'.join(modality_pair)
            fused = th.cat((agg_experts[mod0], agg_experts[mod1]),
                           1)  # -> B x 2D
            temp = self.g_reason_1(fused)  # B x 2D -> B x D
            temp = self.coll_g_dropout(temp)
            temp = self.g_reason_2(F.relu(temp))  # B x D -> B x D
            temp_dict[new_key] = temp
            avail = (ind[mod0].float() * ind[mod1].float()).to(device)
            avai_dict[new_key] = avail

        # Combine the paired features into a mask through elementwise summation
        for mm in temp_dict:
          curr_mask += temp_dict[mm] * avai_dict[mm].unsqueeze(1)
          mask_num += avai_dict[mm]

        curr_mask = th.div(curr_mask, (mask_num + 0.00000000001).unsqueeze(1))
        curr_mask = self.f_reason_1(curr_mask)
        curr_mask = self.coll_f_dropout(curr_mask)
        curr_mask = self.f_reason_2(F.relu(curr_mask))
        masks[modality] = curr_mask

        mod_gu = self.video_GU[modality]
        experts[modality] = mod_gu(experts[modality], masks[modality])

    # If Bert architecture is employed
    if self.vid_cont == 'bert':
      # 0=[CLS] 1=[SEP] 2=[AGG] 3=[MAXP] 4=[MNP] 5=[VLAD] 6=[FEA]
      input_ids_list = []
      token_type_ids_list = []  # Modality id
      # Position (0 = no position, 1 = unknown, >1 = valid position)
      position_ids_list = []
      features_list = []  # Semantics
      attention_mask_list = []  # Valid token or not

      modality_to_tok_map = collections.OrderedDict()

      # [CLS] token
      tok_id = 0
      ids_size = (b,)
      input_ids_list.append(th.full(ids_size, 0, dtype=th.long))
      token_type_ids_list.append(th.full(ids_size, 0, dtype=th.long))
      position_ids_list.append(th.full(ids_size, 0, dtype=th.long).to(device))
      features_list.append(
          th.full((b, self.same_dim), 0, dtype=th.float).to(device))
      attention_mask_list.append(th.full(ids_size, 1, dtype=th.long).to(device))

      # Number of temporal tokens per modality
      if self.vid_inp in ['temp', 'both', 'all']:
        max_expert_tokens = collections.OrderedDict()
        for _, modality in enumerate(self.modalities):
          max_expert_tokens[modality] = experts_feats[modality].size()[1]

      # Make the features_t and raw_captions_t start at the minimum value
      if self.pos_enc == 'tint':

        # Clamp the position encoding to [0, max_position_embedding - 1]
        max_pos = self.vid_bert_params['max_position_embeddings'] - 1
        for _, modality in enumerate(self.modalities):
          experts_feats_t[modality].clamp_(min=0, max=max_pos)
          experts_feats_t[modality] = experts_feats_t[modality].long().to(
              device)

      for _, modality in enumerate(self.modalities):
        token_type = self.expert_dims[modality]['idx']

        # Add an aggregated feature token
        if self.vid_inp in ['agg', 'both', 'all']:
          tok_id += 1
          modality_to_tok_map[modality] = tok_id
          input_ids_list.append(th.full(ids_size, 2, dtype=th.long))
          token_type_ids_list.append(
              th.full(ids_size, token_type, dtype=th.long))
          position_ids_list.append(
              th.full(ids_size, 0, dtype=th.long).to(device))
          if self.out_tok == 'sep':
            features_list.append(
                th.full((b, self.same_dim), 0, dtype=th.float).to(device))
          elif self.out_tok == 'mxp':
            features_list.append(maxp_experts[modality])
          elif self.out_tok == 'mnp':
            features_list.append(mnp_experts[modality])
          attention_mask_list.append(ind[modality].to(dtype=th.long).to(device))
        if self.vid_inp in ['temp', 'both', 'all']:
          for frame_id in range(max_expert_tokens[modality]):
            if self.pos_enc == 'ordr':
              position_ids_list.append(
                  th.full(ids_size, frame_id + 1, dtype=th.long).to(device))
            elif self.pos_enc == 'tint':
              position_ids_list.append(experts_feats_t[modality][:, frame_id])
            elif self.pos_enc == 'type':
              position_ids_list.append(
                  th.full(ids_size, 1, dtype=th.long).to(device))
            tok_id += 1
            input_ids_list.append(th.full(ids_size, 6, dtype=th.long))
            token_type_ids_list.append(
                th.full(ids_size, token_type, dtype=th.long))
            features_list.append(experts_feats[modality][:, frame_id, :])
            attention_mask_list.append(
                experts_feats_ind[modality][:, frame_id].to(dtype=th.long))

      features = th.stack(features_list, dim=1).to(self.device)
      input_ids = th.stack(input_ids_list, dim=1).to(self.device)
      token_type_ids = th.stack(token_type_ids_list, dim=1).to(self.device)
      if self.pos_enc != 'none':
        position_ids = th.stack(position_ids_list, dim=1).to(self.device)
      else:
        position_ids = None
      attention_mask = th.stack(attention_mask_list, dim=1).to(self.device)

      if self.debug_dataloader and debug:
        token_ids = token_ids.view(b, captions_per_video, max_text_words,
                                   feat_dim)
        self.display_minibatch(token_ids, input_ids, attention_mask,
                               token_type_ids, position_ids, features)
        token_ids = token_ids.view(b * captions_per_video, max_text_words,
                                   feat_dim)

      vid_bert_output = self.vid_bert(input_ids,
                                      attention_mask=attention_mask,
                                      token_type_ids=token_type_ids,
                                      position_ids=position_ids,
                                      features=features)

      last_layer = vid_bert_output[0]
      vid_embd = last_layer[:, 0]

      for _, modality in enumerate(self.modalities):
        experts[modality] = last_layer[:, modality_to_tok_map[modality]]

    if self.vid_wgh == 'nrm':
      vid_weights = self.compute_weights_from_norm(experts)
    elif self.vid_wgh == 'emb':
      vid_weights = self.compute_weights_from_emb(vid_embd)
    elif self.vid_wgh == 'none':
      vid_weights = th.ones(b, m).to(device)
    else:
      msg = 'video weighting mode {} not supported'
      raise NotImplementedError(msg.format(self.vid_wgh))

    if not self.keep_missing_modalities:
      # Zero padding of the missing modalities
      available = th.zeros(b, m).to(device)
      for idx, mod in enumerate(self.modalities):
        available[:, idx] = ind[mod].float()  # B x M
      vid_weights = vid_weights * available

    # Normalize video weights so that they sum to one
    vid_weights = nn.functional.normalize(vid_weights, p=1, dim=-1)

    if self.txt_wgh == 'emb':
      text_weights = self.compute_weights_from_emb(text)
    elif self.txt_wgh == 'none':
      text_weights = th.ones(b, captions_per_video, m).to(device)
    else:
      msg = 'txt weighting mode {} not supported'
      raise NotImplementedError(msg.format(self.txt_wgh))

    # Normalize text weights so that they sum to one
    text_weights = nn.functional.normalize(text_weights, p=1, dim=-1)

    # L2 Normalization of each expert
    if self.normalize_experts:
      for _, modality in enumerate(self.modalities):
        experts[modality] = nn.functional.normalize(experts[modality], dim=-1)
        text_embd[modality] = nn.functional.normalize(text_embd[modality],
                                                      dim=-1)

    if self.training:
      merge_caption_similiarities = 'avg'
    else:
      merge_caption_similiarities = self.test_caption_mode
    self.merge_caption_similarities = merge_caption_similiarities

    if out == 'conf':  # Output confusion matrix
      cross_view_conf_matrix = sharded_cross_view_inner_product(
          vid_embds=experts,
          text_embds=text_embd,
          vid_weights=vid_weights,
          text_weights=text_weights,
          subspaces=self.modalities,
          merge_caption_similiarities=merge_caption_similiarities,
      )
      return {
          'modalities': self.modalities,
          'cross_view_conf_matrix': cross_view_conf_matrix,
      }
    else:  # Output the embeddings
      # Transform the dictionaries into tensors
      vid_embds_list = []
      text_embds_list = []
      for idx, mod in enumerate(self.modalities):
        vid_embds_list.append(experts[mod].unsqueeze(1))
        text_embds_list.append(text_embd[mod].unsqueeze(1))
      vid_embds = th.cat(vid_embds_list, 1)
      text_embds = th.cat(text_embds_list, 1)

      return {
          'vid_embds': vid_embds,
          'text_embds': text_embds,
          'vid_weights': vid_weights,
          'text_weights': text_weights,
      }

  def display_minibatch(self, token_ids, input_ids, attention_mask,
                        token_type_ids, position_ids, features):
    for i in range(1):
      logger.debug()
      # logger.debug(f'Sample {i}:')
      logger.debug('Text:')
      ids = token_ids[i, 0, :, 0].cpu().numpy()
      logger.debug(ids)

      tokens = self.tokenizer.convert_ids_to_tokens(ids)
      logger.debug(tokens)

      logger.debug('Video:')
      # logger.debug(f'input_ids: {input_ids[i]}')
      # logger.debug(f'attention_mask: {attention_mask[i]}')
      # logger.debug(f'token_type_ids: {token_type_ids[i]}')
      # logger.debug(f'position_ids: {position_ids[i]}')
      logger.debug(features[i].shape)


class GatedEmbeddingUnit(nn.Module):
  """Gated embedding module.

  as described in
  "Learning a Text-Video Embedding from Incomplete and Heterogeneous Data"
  """

  def __init__(self, input_dimension, output_dimension, use_bn, normalize):
    super(GatedEmbeddingUnit, self).__init__()

    self.fc = nn.Linear(input_dimension, output_dimension)
    self.cg = ContextGating(output_dimension, add_batch_norm=use_bn)
    self.normalize = normalize

  def forward(self, x):
    x = self.fc(x)
    x = self.cg(x)
    if self.normalize:
      x = F.normalize(x, dim=-1)
    return x


class MimicCEGatedEmbeddingUnit(nn.Module):

  def __init__(self, input_dimension, output_dimension, use_bn):
    super().__init__()
    self.cg = ContextGating(input_dimension, add_batch_norm=use_bn)

  def forward(self, x):
    x = self.cg(x)
    x = F.normalize(x, dim=-1)
    return x


class ReduceDim(nn.Module):

  def __init__(self, input_dimension, output_dimension):
    super(ReduceDim, self).__init__()
    self.fc = nn.Linear(input_dimension, output_dimension)

  def forward(self, x):
    x = self.fc(x)
    x = F.normalize(x, dim=-1)
    return x


class GatedLinearUnit(nn.Module):

  def forward(self, x, mask):
    x = th.cat((x, mask), 1)
    return F.glu(x, 1)


class ContextGating(nn.Module):
  """Context gating class."""

  def __init__(self, dimension, add_batch_norm=True):
    super(ContextGating, self).__init__()
    self.fc = nn.Linear(dimension, dimension)
    self.add_batch_norm = add_batch_norm
    self.batch_norm = nn.BatchNorm1d(dimension)

  def forward(self, x):
    x1 = self.fc(x)
    if self.add_batch_norm:
      x1 = self.batch_norm(x1)
    x = th.cat((x, x1), 1)
    return F.glu(x, 1)


class GatedEmbeddingUnitReasoning(nn.Module):

  def __init__(self, output_dimension):
    super(GatedEmbeddingUnitReasoning, self).__init__()
    self.cg = ContextGatingReasoning(output_dimension)

  def forward(self, x, mask):
    x = self.cg(x, mask)
    x = F.normalize(x, dim=-1)
    return x


class ContextGatingReasoning(nn.Module):
  """Context gating reasoning class."""

  def __init__(self, dimension, add_batch_norm=True):
    super(ContextGatingReasoning, self).__init__()
    self.fc = nn.Linear(dimension, dimension)
    self.add_batch_norm = add_batch_norm
    self.batch_norm = nn.BatchNorm1d(dimension)
    self.batch_norm2 = nn.BatchNorm1d(dimension)

  def forward(self, x, x1):

    x2 = self.fc(x)

    if self.add_batch_norm:
      x1 = self.batch_norm(x1)
      x2 = self.batch_norm2(x2)

    t = x1 + x2

    x = th.cat((x, t), 1)
    return F.glu(x, 1)


def sharded_cross_view_inner_product(vid_embds,
                                     text_embds,
                                     vid_weights,
                                     text_weights,
                                     subspaces,
                                     merge_caption_similiarities='avg'):
  """Compute similarities between all captions and videos."""

  b = vid_embds[subspaces[0]].size(0)
  device = vid_embds[subspaces[0]].device
  num_caps = text_embds[subspaces[0]].size(1)
  m = len(subspaces)

  # unroll separate captions onto first dimension and treat them separately
  sims = th.zeros(b * num_caps, b, device=device)

  text_weights = text_weights.view(b * num_caps, -1)
  vid_weights = vid_weights.view(b, -1)

  moe_weights = vid_weights[None, :, :] * text_weights[:, None, :]

  norm_weights = th.sum(moe_weights, dim=2)
  norm_weights = norm_weights.unsqueeze(2)
  # If only one modality is used and is missing in some videos, moe_weights will
  # be zero.
  # To avoid division by zero, replace zeros by epsilon
  # (or anything else, in that case moe_weights are zero anyway)
  norm_weights[norm_weights == 0] = 1E-5
  moe_weights = th.div(moe_weights, norm_weights)

  assert list(moe_weights.size()) == [b * num_caps, b, m]

  for idx, mod in enumerate(subspaces):
    text_embds[mod] = text_embds[mod].view(b * num_caps, -1)
    sims += moe_weights[:, :, idx] * th.matmul(text_embds[mod],
                                               vid_embds[mod].t())

  if num_caps > 1:
    # aggregate similarities from different captions
    if merge_caption_similiarities == 'avg':
      sims = sims.view(b, num_caps, b)
      sims = th.mean(sims, dim=1)
      sims = sims.view(b, b)
    elif merge_caption_similiarities == 'indep':
      pass
    else:
      msg = 'unrecognised merge mode: {}'
      raise ValueError(msg.format(merge_caption_similiarities))
  return sims
