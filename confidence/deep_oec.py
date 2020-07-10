from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import ipdb
import sys
from confidence.oec import OEC, prepare_oec_input
import itertools

def _oec_fpass(oec, oec_input):
    n_way, n_query, _ = oec_input.shape
    oec_output = oec(oec_input.view(-1, oec_input.shape[-1]))
    oec_output = oec_output.view(n_way,n_query,-1)
    return oec_output

class DeepOECConfidence(nn.Module):
  def __init__(self, oecs, fsmodel, init_sample, opt):
    super(DeepOECConfidence, self).__init__()
    self.fsmodel = fsmodel
    self.oec_depth_aggregation = opt['oec_depth_aggregation']
    self.embed_in_eval = opt['oec_embed_in_eval']
    if oecs is not None: # pretrained oec
        self.oecs = oecs
    else: 
        if opt['oec_depth'] == 'penult':
            self.ls = [self.fsmodel.encoder.depth - 1]
        elif opt['oec_depth'] == 'all':
            self.ls = list(range(self.fsmodel.encoder.depth))
        else:
            raise ValueError("invalid `oec_depth`")
        if self.oec_depth_aggregation == 'input':
            in_inp = self._deep_backbone_fpass(init_sample['xs'], init_sample['xq'], self.ls)
            idim = in_inp.shape[-1]
            hidden_layers = [] if opt['arch']=='-' else list(map(int, opt['arch'].split(',')))
            self.oecs = [OEC([idim]+ hidden_layers +[1],
                        use_residual=opt['residual'],
                        track_running_stats=opt['oec_track_running_stats'],
                        norm_type=opt['oec_norm_type'])]
        elif self.oec_depth_aggregation in ['output-sum','output-lse']:
            self.oecs = []
            for l in self.ls:
                in_inp = self._deep_backbone_fpass(init_sample['xs'], init_sample['xq'], [l])
                idim = in_inp.shape[-1]
                hidden_layers = [] if opt['arch']=='-' else list(map(int, opt['arch'].split(',')))
                self.oecs.append(OEC([idim]+ hidden_layers +[1],
                            use_residual=opt['residual'],
                            track_running_stats=opt['oec_track_running_stats'],
                            norm_type=opt['oec_norm_type']))
        else:
            raise ValueError("invalid `oec_depth_aggregation`")
    for n, oec in enumerate(self.oecs):
        self.add_module(f"oec_{n}", oec)
        
    

  def support(self, s):
    self.s = s

  def confidence_parameters(self):
    return itertools.chain(*[oec.parameters() for oec in self.oecs])
  def pretrain_parameters(self):
    return itertools.chain(*[oec.residual_layer.parameters() for oec in self.oecs])

  def score(self, x, detach=True):
    s = self.s
    if len(x.shape) == 4:
        x = x.unsqueeze(0)
    if self.oec_depth_aggregation == 'input':
        oec_input = self._deep_backbone_fpass(s, x, self.ls)
        score_c = _oec_fpass(self.oecs[0], oec_input)
        

    elif self.oec_depth_aggregation in ['output-sum','output-lse']:
        score_l_c = []
        for l, oec in zip(self.ls, self.oecs):
            oec_input = self._deep_backbone_fpass(s, x, [l])
            score_l_c.append(_oec_fpass(oec, oec_input))
        if self.oec_depth_aggregation == 'output-sum':
            score_c = torch.stack(score_l_c).sum(0)
        else:
            score_c = torch.logsumexp(torch.stack(score_l_c), 0)

    score = score_c.max(0)[0].squeeze()      
    return score if not detach else score.detach()

  def _deep_backbone_fpass(self, xs, xq, ls):
    if self.embed_in_eval:
        self.fsmodel.eval()
    assert 5 == len(xs.shape) == len(xq.shape)
    # Prepare OEC input
    # _log_p_y = self.fsmodel.log_p_y(xs, xq, no_grad=True, mask=None)
    # 
    def enc(x):
        shape = x.shape
        x = x.view(-1, shape[-3],shape[-2],shape[-1])
        z = torch.cat([self.fsmodel.encoder.intermediate_forward(x, l, avg_pool=True).detach() for l in ls],-1)
        return z.view(shape[0], shape[1], -1)

    zq = enc(xq)
    zq = zq.view(-1, zq.size(-1))
    zs = enc(xs)
    oec_inp = prepare_oec_input(zs, zq, repeat_query=True) # (n_way, n_way*n_q, Z)
    return oec_inp.detach()
    
