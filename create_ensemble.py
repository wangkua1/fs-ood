import argparse
import torch
from classify_classic  import ResNetClassifier
import numpy as np
import ipdb
import os
from collections import defaultdict, OrderedDict

class Ensemble(torch.nn.Module):
  def __init__(self, nets, aggr='mixture'):
    super(Ensemble, self).__init__()
    self.nets = nets
    self.aggr = aggr

  def forward(self, x):
    y = torch.stack([n(x) for n in self.nets])
    if self.aggr == 'mixture':
      ret = torch.logsumexp(y, 0)
    if self.aggr == 'poe':
      ret = torch.mean(y, 0)
    return ret
  
  def support(self, s):
    for n in self.nets:
      n.support(s)

  def eval(self):
    for n in self.nets:
      n.eval()

  def train(self):
    for n in self.nets:
      n.train()

  def score(self, x):
    return torch.stack([n.score(x) for n in self.nets]).mean(0)
    
