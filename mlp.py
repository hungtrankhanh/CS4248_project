import torch
import torch.nn as nn
import torch.nn.functional as F


class IntervenionModel(nn.Module):
  def __init__(self, D_in, H1, D_out = 1):
      super().__init__()
      self.linear1 = nn.Linear(D_in, H1)
      self.linear2 = nn.Linear(H1, D_out)
      self.classify = nn.Sigmoid()

  def forward(self, x):
      x = F.relu(self.linear1(x))
      x = self.linear2(x)
      return  self.classify(x)
