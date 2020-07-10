from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import ipdb
import sys
sys.path.append('..')
from utils import euclidean_dist


class DKDEConfidence:
    def __init__(self, encoder):
        self.encoder = encoder 
        
    def support(self,s):
        imdim = s.shape[2:]
        self.s = s.view(-1,*imdim)

    def score(self, x):
        preds = []
        for l in range(self.encoder.depth):
          enc = lambda x: self.encoder.intermediate_forward(x, l, avg_pool=True).view(x.size(0),-1)
          zq = enc(x)
          zs = enc(self.s)

          preds.append(-euclidean_dist(zs, zq)) # (S, N)
        preds = torch.stack(preds) # (L, C, N)
        preds = preds.sum(0).t() # (N, C)
        return preds.max(1)[0]
        
        
