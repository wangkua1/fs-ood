from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import ipdb
import sys
from .lcbo import OEC



def prepare_lcbo_input(zs, zq, repeat_query=False):
    """
    Input
        zs: the supports (n_way, n_shot, zdim)
        zq: the queries (n_way*n_query, zdim)
    Output
        (n_way*n_query, zdim*3)
    """
    n_way, _,  zdim = zs.shape
    _inp0 = zs.mean(1)

    if repeat_query: # For eval
        _inp1 = zq.unsqueeze(0).repeat(n_way, 1, 1)
    else: # For Training
        _inp1 = zq.view(n_way, -1, zdim)  # (n_way, n_query, zdim)

    # Repeat prototypes n_query or n_query*n_way times
    _inp0 = _inp0.unsqueeze(1).repeat(1, _inp1.shape[1], 1)
    # Concat
    inp = torch.cat([_inp0, _inp1], -1)  # (n_way, n_query, zdim*3)
    return inp


def deep_backbone_fpass(xs, xq, model, repeat_query=False):
    assert 5 == len(xs.shape) == len(xq.shape)
    # Prepare OEC input
    _log_p_y = model.log_p_y(xs, xq, no_grad=True, mask=None)
    # 
    def enc(x):
        shape = x.shape
        x = x.view(-1, shape[-3],shape[-2],shape[-1])
        z = torch.cat([model.encoder.intermediate_forward(x, l, avg_pool=True).detach() for l in range(model.encoder.depth)],-1)
        return z.view(shape[0], shape[1], -1)
    zq = enc(xq)
    zq = zq.view(-1, zq.size(-1))
    zs = enc(xs)
    lcbo_inp = prepare_lcbo_input(zs, zq, repeat_query=repeat_query)
    return lcbo_inp.detach()


def get_score(s, x, model, lcbo):
    if len(x.shape) == 4:
        x = x.unsqueeze(0)
    # In-query
    lcbo_input = deep_backbone_fpass(s, x, model, repeat_query=True)
    # OEC Forward pass
    n_way, n_query, _ = lcbo_input.shape
    lcbo_output = lcbo(lcbo_input.view(-1, lcbo_input.shape[-1]))
    N = lcbo_output.shape[0]
    lcbo_output = lcbo_output.view(n_way,n_query,-1)
    return lcbo_output.max(0)[0]


class DeepOECConfidence(nn.Module):
  def __init__(self, lcbo, fsmodel):
    super(DeepOECConfidence, self).__init__()

    self.lcbo = lcbo 
    self.fsmodel = fsmodel

  def support(self, s):
    self.s = s

  def confidence_parameters(self):
    return self.lcbo.parameters()

  def score(self, x, detach=True):
    s = get_score(self.s, x, self.fsmodel, self.lcbo).squeeze()
    return s if not detach else s.detach()