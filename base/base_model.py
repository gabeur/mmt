"""Models base class."""
import abc

import numpy as np
import torch.nn as nn


class BaseModel(nn.Module):
  """Base class for all models."""

  @abc.abstractmethod
  def forward(self, *inputs):
    """Forward pass logic."""
    raise NotImplementedError

  def __str__(self):
    """Model prints with number of trainable parameters."""
    model_parameters = filter(lambda p: p.requires_grad, self.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return super().__str__() + f"\nTrainable parameters: {params}"
