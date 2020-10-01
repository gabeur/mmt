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
"""Logic to embed language tokens."""
import logging
import os

from gensim.models.keyedvectors import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import torch
from torch import nn

logger = logging.getLogger(__name__)


class TxtEmbeddings(nn.Module):
  """Construct the embeddings from word, position and token_type embeddings."""

  def __init__(self, vocab_size=None, emb_dim=None, ckpt=None, freeze=False):
    super(TxtEmbeddings, self).__init__()

    if ckpt is not None:
      if isinstance(ckpt, str):
        logger.debug('Loading the pretrained word embeddings from %s ...', ckpt)
        pretrained_dict = torch.load(ckpt)
        weight = pretrained_dict['bert.embeddings.word_embeddings.weight']

      elif isinstance(ckpt, torch.FloatTensor):
        weight = ckpt

      self.nb_words = weight.size()[0]
      logger.debug('Nb of words in the embedding table: %d', self.nb_words)
      self.text_dim = weight.size()[1]
      self.word_embeddings = nn.Embedding.from_pretrained(weight,
                                                          freeze=freeze,
                                                          padding_idx=0)

    else:
      # padding_idx=0 means the first row will be set at zeros and will stay so
      # (zero gradients). To be used for padding.
      self.word_embeddings = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
      self.text_dim = emb_dim

      if freeze:
        model = self.word_embeddings
        for param in model.parameters():
          param.requires_grad = False

  def forward(self, input_ids=None):
    inputs_embeds = self.word_embeddings(input_ids)
    return inputs_embeds


class WeTokenizer():
  """Word embeddings tokenizer."""

  def __init__(self, we_filepath, freeze=False):
    if we_filepath.endswith('.bin'):
      binary = True
      self.we = KeyedVectors.load_word2vec_format(we_filepath, binary=binary)
    elif we_filepath.endswith('.txt'):
      w2v_format_path = we_filepath.replace('.txt', '.w2v')
      if not os.path.exists(w2v_format_path):
        # Convert to right format
        glove2word2vec(we_filepath, w2v_format_path)
      self.we = KeyedVectors.load_word2vec_format(w2v_format_path, binary=False)

    self.text_dim = self.we.vectors.shape[1]

    # Add a line of zeros corresponding to the padding token and unknown token
    pad_vec = torch.zeros((2, self.text_dim))
    raw_table = torch.FloatTensor(self.we.vectors)
    self.weights = torch.cat((pad_vec, raw_table))

    # Add the padding token
    self.words = ['[PAD]', '[UNK]'] + list(self.we.vocab.keys())

    self.we_model = TxtEmbeddings(ckpt=self.weights, freeze=freeze)

  def tokenize(self, text):
    """Convert a text into tokens."""
    # Un-capitalize
    text = text.lower()

    # Split the text into words
    words = text.split(' ')

    # Remove special characters from words
    words = [''.join(e for e in word if e.isalnum()) for word in words]

    # Remove the words out of vocabulary
    words = [word for word in words if word in self.words]

    if not words:
      words = ['[UNK]']

    return words

  def convert_tokens_to_ids(self, tokens):
    return [self.words.index(token) for token in tokens]

  def convert_ids_to_tokens(self, ids):
    return [self.words[idx] for idx in ids]
