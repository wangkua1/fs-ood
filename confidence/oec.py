from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import ipdb
import sys
sys.path.append('..')
from utils import euclidean_dist


class OEC(nn.Module):
    """
        The idea:
            A binary classifier network that does this
            p(query = outlier | class )
            ** here we are not conditioned on the `task', but
               rather on one `class'
        Motivation:
            The world is imperfect, so is your acquisition function
            This is a way to learn the acquisition function
        Attempt1:
            Simple takes as input a concatenation of
            [proto, proto_var, query]
            and feed it through a fcNN
    """
    def __init__(self, net=[64*3,100,100,1], use_residual=False, track_running_stats=True, norm_type='bn'):
        super(OEC, self).__init__()
        norm = {
            'bn': lambda d: nn.BatchNorm1d(d, track_running_stats=track_running_stats),
            'in': lambda d: nn.GroupNorm(d,d), #Instance Norm
            'gn': lambda d: nn.GroupNorm(d//2,d)
        }[norm_type]
        def _layers(dims):
            layers = []
            for idx in range(len(dims)-1):
                L = nn.Linear(dims[idx], dims[idx+1])
                layers.append(L)
                layers.append(nn.ReLU())
                layers.append(norm(dims[idx+1]))
            return layers

        self.net = nn.Sequential(
                *_layers(net)[:-2] # ** [:-2] to exclude BN, ReLU
            )
        self.use_residual = use_residual
        self.residual_layer = nn.Linear(1,1)

    def forward(self, x):
        """
            x: Tensor of shape (batch, dim)
        """
        z = self.net(x).squeeze()

        if self.use_residual:
            zdim = x.shape[-1] // 2
            residual = (x[:,:zdim] - x[:,-zdim:])
            z += self.residual_layer(-residual.pow(2).mean(-1,keepdim=True)).squeeze()

        return z




def prepare_oec_input(zs, zq, repeat_query=False):
    """
    Input
        zs: the supports (n_way, n_shot, zdim)
        zq: the queries (n_way*n_query, zdim)
    Output
        (n_way, n_way*n_query, zdim*3)
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


def backbone_fpass(xs, xq, model, repeat_query=False):
    # Prepare OEC input
    _log_p_y = model.log_p_y(xs, xq, no_grad=True, mask=None)
    zq, zs = _log_p_y['zq'], _log_p_y['supports']
    oec_inp = prepare_oec_input(zs, zq, repeat_query=repeat_query)
    return oec_inp.detach()


def get_score(s, x, model, oec):
    if len(x.shape) == 4:
        x = x.unsqueeze(0)
    # In-query
    oec_input = backbone_fpass(s, x, model, repeat_query=True)
    # OEC Forward pass
    n_way, n_query, _ = oec_input.shape
    oec_output = oec(oec_input.view(-1, oec_input.shape[-1]))
    N = oec_output.shape[0]
    oec_output = oec_output.view(n_way,n_query,-1)
    return oec_output.max(0)[0]
    

class OECConfidence(nn.Module):
  def __init__(self, oec, fsmodel, init_sample, opt):
    super(OECConfidence, self).__init__()

    if oec is not None: # pretrained oec
        self.oec = oec 
    else: 
        in_inp = backbone_fpass(init_sample['xs'], init_sample['xq'], fsmodel)
        idim = in_inp.shape[-1]
        hidden_layers = [] if opt['arch']=='-' else list(map(int, opt['arch'].split(',')))
        self.oec = OEC([idim]+hidden_layers+[1],
                    use_residual=opt['residual'],
                    track_running_stats=opt['oec_track_running_stats'],
                    norm_type=opt['oec_norm_type'])

    self.fsmodel = fsmodel

  def confidence_parameters(self):
    return self.oec.parameters()

  def pretrain_parameters(self):
    return self.oec.residual_layer.parameters()

  def support(self, s):
    self.s = s

  def score(self, x, detach=True):
    # Make sure fs_model is always in eval mode
    self.fsmodel.eval()
    s = get_score(self.s, x, self.fsmodel, self.oec).squeeze()
    return s if not detach else s.detach()


        
        
