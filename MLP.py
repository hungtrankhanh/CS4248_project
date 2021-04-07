import torch
import torch.nn as nn
class IntervenionModel(nn.Module):
  def __init__(self, D_in, H1, D_out = 1):
      super().__init__()
      self.linear1 = nn.Linear(D_in, H1)
      # self.weighs_init(self.linear1)
      self.linear2 = nn.Linear(H1, D_out)
      # self.weighs_init(self.linear2)
      self.classify = nn.Sigmoid()

  # def weighs_init(self,m):
  #     n = m.in_features
  #     y = 1.0/np.sqrt(n)
  #     m.weight.data.uniform_(-y , y)
  #     m.bias.data.fill_(0.0)

  def forward(self, x):
      x = F.relu(self.linear1(x))
      x = self.linear2(x)
      return  self.classify(x)
