from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import ipdb
import itertools
from tqdm import tqdm
from anomaly import load_ood_data, show_ood_detection_results_softmax
import sklearn.covariance
import os
from collections import defaultdict, OrderedDict
import matplotlib.pylab as plt

from functools import partial
import pandas
from copy import deepcopy
from data import get_dataset, get_transform, load_episode
from create_ensemble import Ensemble
from glow.model import Glow
import cv2
import torchvision
import torchvision.utils as vutils
from torch.autograd import grad, Variable


def fit_precision(z, type='full'):
    if type == 'full':
      cov_obj = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
    if  type == 'shrunk':
      cov_obj = sklearn.covariance.ShrunkCovariance(assume_centered=False)
    #             
    cov_obj.fit(z.cpu().numpy())
    temp_precision = torch.from_numpy(cov_obj.precision_).float().cuda()
    # ## 
    cov = torch.mm(z.t(), z)
    U, D, V = np.linalg.svd(cov.detach().cpu().numpy(), compute_uv=True, full_matrices=False)
    ## np check
    aa = cov_obj.covariance_
    B = cov_obj.precision_
    ee = np.dot(B, aa)
    close = np.allclose(aa, np.dot(aa, np.dot(B, aa))) # this is true, but  np.dot(B,aa) isn't eye()
    return  temp_precision

def fit_mahala(xs, tr, f_enc, device, share_cov=False,cov_type='full'):

    def _fit_precision(z):
      if cov_type=='full':
        return fit_precision(z).to(z.device)
      if cov_type=='shrunk':
        return fit_precision(z, type='shrunk').to(z.device)
      if cov_type=='diag':
        return 1./ (torch.diag(z.std(0))+1e-10)
      if cov_type=='iso':
        return torch.eye(z.shape[1]).to(z.device)
      raise
    mus = []
    precisions = []
    if share_cov:
      diffs = []
    with torch.no_grad():
      for cx in xs:
        if tr:
          cx = torch.stack([tr(x_) for x_ in cx])
        # print("encoding")
        z = f_enc(cx.to(device))    
        mus.append(torch.mean(z, 0))
        if not share_cov:
          # print("fitting precision")
          precisions.append(_fit_precision(z))
        else:
          diffs.append( z - torch.mean(z, 0))
    if share_cov:
      # print("fitting precision")
      precisions.append(_fit_precision(torch.cat(diffs, 0)))

    return mus, precisions

def gaussian_logp(z, mu, precision):
  diff = z - mu
  c = -0.5*torch.mm(torch.mm(diff, precision), diff.t()).diag()
  return c

class DeepMahala(object):
  def __init__(self, x_train, y_train, tr, encoder, device, pretrained_path='', num_feats = 5, num_classes=10,fit=True, normalize=None):
    super(DeepMahala, self).__init__()
    self.num_feats = num_feats
    self.num_classes = num_classes
    self.normalize = normalize
    if os.path.exists(pretrained_path):
      self.mu_l_c, self.precision_l, self.encoder = torch.load(pretrained_path)
    else:
      self.mu_l_c = [] # mu_{layer, class}
      self.precision_l = [] # [ p0, ..., p4] shape [64x64, ..., 512x512]
      self.encoder = encoder
      if fit:
          for l in range(self.num_feats):
            xs = [x_train[y_train == c] for c in range(num_classes)]
            enc = lambda x: self.encoder.intermediate_forward(x, l, avg_pool=True)
            mus, precs = fit_mahala(xs, tr, enc, device, share_cov=True)
            self.mu_l_c.append(mus)
            self.precision_l.append(precs[0])

      if pretrained_path != '':
        torch.save((self.mu_l_c, self.precision_l, self.encoder), pretrained_path)
    
  def predict(self, x, ls=[0,1,2,3,4],reduction='', mu_l_c=None, precision_l=None, avg_pool=True, use_both=False, weights=None, g_magnitude=0):
    def _run(mu_l_c, precision_l, x):
      preds = []
      for l in ls:
        enc = lambda x: self.encoder.intermediate_forward(x, l, avg_pool=avg_pool).view(x.size(0),-1)
        z = enc(x)
        s_l = []
        for c in range(len(mu_l_c[l])):
          s = gaussian_logp(z, mu_l_c[l][c], precision_l[l])
          s_l.append(s)
        preds.append(torch.stack(s_l))
      preds = torch.stack(preds) # (L, C, N)
      if weights is not None:
        preds = (weights * preds.permute(1,2,0)).sum(-1) / torch.sum(weights)
      else:
        preds = preds.mean(0)
      preds = preds.t() # (N, C)
      return preds
      
    if not mu_l_c:
      mu_l_c = self.mu_l_c
      precision_l = self.precision_l

    x = Variable(x)
    x.requires_grad_()
    preds = _run(mu_l_c, precision_l, x)
    if g_magnitude > 0:
      # 1. select predicted prob
      pyx = preds.max(1)[0]
      # 2. grad
      g = grad(torch.mean(-pyx), [x])[0]
      # 3. fgsm
      g = torch.ge(g.data, 0)
      g = (g.float() - .5) * 2
      # 4. normalize to data range
      std = torch.FloatTensor(self.normalize.std).to(x.device).view(1,3,1,1)
      g = g / std 
      # 5. re-run with x+g
      preds = _run(mu_l_c, precision_l, x - g_magnitude * g)

    if use_both:
      raise # not been updated....
      # preds2 = _run(self.mu_l_c, self.precision_l)
      # preds = torch.cat([preds, preds2],-1)

    if reduction == 'max':
      return preds.max(1)[0]
    elif reduction == 'mean':
      return preds.mean(1)
    else:
      return preds

class DMConfidence(nn.Module):
    def __init__(self, dm_obj, predict_kw, use_support, episodic_cov_type='shrunk'):
        super(DMConfidence, self).__init__()
        self.dm_obj = dm_obj
        self.encoder = dm_obj.encoder
        self.predict_kw = predict_kw
        self.use_support = use_support
        self.episodic_cov_type = episodic_cov_type
        self.weights = nn.Parameter(torch.ones(self.dm_obj.num_feats))
        self.predict_kw['weights'] = self.weights

    def support(self,s):
        if self.use_support:
            # compute episodic mu_l_c and precision_l
            ls = self.predict_kw['ls']
            mu_l_c = {}
            precision_l = {}
            for l in ls:
              f_enc = lambda x: self.encoder.intermediate_forward(x, l, True).view(x.size(0),-1)
              mu_c, p_c = fit_mahala(s,  None, f_enc, s.device, share_cov=True, cov_type=self.episodic_cov_type)
              mu_l_c[l] = mu_c
              precision_l[l] = p_c[0]
            self.predict_kw['mu_l_c'] = mu_l_c
            self.predict_kw['precision_l'] = precision_l


    def confidence_parameters(self):
      return [self.weights]

    def pretrain_parameters(self):
      return None

    def score(self, x, detach=False):
        c, h, w = x.shape[-3:]
        s = self.dm_obj.predict(x.view(-1, c, h, w), **self.predict_kw)
        return s.detach() if detach else s
        
        
