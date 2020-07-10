import ipdb
import copy
from functools import partial
from collections import defaultdict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# Local imports
import backbone
from utils import euclidean_dist

class Protonet(nn.Module):
    def __init__(self, encoder, f_name='dummy', x_dim=[1,28,28], method='', oe_lambda=0,
                 ooe_lambda=0, lcbo_arch=[500,500], lcbo_aggregation='max'):
        super(Protonet, self).__init__()

        self.encoder = encoder
        self.ooe_lambda = ooe_lambda
        self.oe_lambda = oe_lambda
        self.decision = 'baseline'
        self.method = method  # Can be: baseline, outlier_exposure, ...

    def log_p_y(self, xs, xq, no_grad=False, mask=None):
        """log_p_y

        Args:
            xs: support set (#way, #shot, ...)
            xq: support set (#query classes, #queries, ...)
            return_target_inds: set to True only if #way and #query index over the same classes
        Return:
            a dict of {
                'log_p_y': log p(y) (#query classes, #queries, #way)
                'target_inds': ...
                'logits': ...
            }
        """
        ret_dict = {}

        # xs.size() == (50, 3, 1, 28, 28)  for 50-way, 3-shot classification
        # xq.size() == (50, 5, 1, 28, 28)  for 50-way, 5 query points for each of the 50 classes
        n_class = xs.size(0)
        n_query_class = xq.size(0)
        n_support = xs.size(1)
        n_query = xq.size(1)

        # Combines the support and query points into one "minibatch" to get embeddings
        x = torch.cat([xs.reshape(n_class * n_support, *xs.size()[2:]),
                       xq.reshape(n_query_class * n_query, *xq.size()[2:])], 0)

        try:
            z = self.encoder(x, no_grad=no_grad)
        except:
            z = self.encoder(x)  # If the encoder doesn't support option no_grad

        z_dim = z.size(-1)
        supports = z[:n_class*n_support].view(n_class, n_support, z_dim)
        ret_dict['supports'] = supports
        # Prototype representations of the support images in each class: (50, 64)
        if mask is not None:
            mask = mask.unsqueeze(-1)
            # ipdb.set_trace()
            z_proto = (supports * mask).sum(1) / mask.sum(1)
        else:
            z_proto = supports.mean(1)
            z_proto_var = supports.var(1)

        ret_dict['z'] = z  # z.shape == (50, 256)
        # Embeddings of the query points: (500, 64)
        zq = z[n_class*n_support:]
        ret_dict['zq'] = zq

        def _vectorized_mog_logp(zq, z_proto_m, z_proto_var):
            res = []
            for ci, (_m, _v) in enumerate(zip(z_proto_m, z_proto_var)):
                _m, _v = _m.unsqueeze(0), _v.unsqueeze(0)
                _v = _v + 1e-7
                logp = -1 * (torch.log(_v).sum() +
                             (torch.pow(zq - _m , 2) / _v).sum(-1))
                res.append(logp)
            return torch.transpose(torch.stack(res), 0, 1)


        if self.decision == 'baseline':
            # Distances between each query embedding and each class prototype
            dists = euclidean_dist(zq, z_proto)

            # Class logits for each query point and each class: (50, 10, 50) [#query classes, #queries per class, #support classes] ?
            log_p_y = F.log_softmax(-dists, dim=1).view(n_query_class, n_query, -1)
            # Prepare logits
            ret_dict['logits'] = (-dists).view(n_query_class, n_query, -1)
        elif self.decision == 'maha':
            raise "Well, we cannot fit a full-covariance b/c N < D"
            # tied full-covariance
            diff = supports - z_proto.unsqueeze(1) # (way, shot, dim)
            diff = diff.view(-1, diff.shape[-1]) # (way*shot, dim) b/c shared across way
            cov = torch.mean(torch.stack([torch.ger(vec,vec) for vec in diff]),0)
            mahas = []
            for _zq in zq:
                mahas.append()
        elif self.decision == 'tied-var':
            if mask is not None:
                raise # implement the masking part
            z_proto_var = supports.view(-1, supports.shape[-1]).var(0).unsqueeze(0).repeat(5,1)
            log_p_y = _vectorized_mog_logp(zq, z_proto, z_proto_var).view(n_query_class, n_query, -1)
            ret_dict['logits'] = log_p_y

        ret_dict['log_p_y']  = log_p_y
        # Prepare target_inds
        if n_class == n_query_class:
            target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
            target_inds.requires_grad = False
            if xq.is_cuda:
                target_inds = target_inds.cuda()
            ret_dict['target_inds'] = target_inds

        return ret_dict


    def loss(self, sample):
        _log_p_y = self.log_p_y(sample['xs'], sample['xq'])
        log_p_y, target_inds, _logits, _z = _log_p_y['log_p_y'], _log_p_y['target_inds'], _log_p_y['logits'], _log_p_y['z']

        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
        _, y_hat = log_p_y.max(2)
        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()

        ret_state = {
            'acc': acc_val.item(),
            'class_loss': loss_val.item()
        }

        if self.method == 'outlier_exposure':
            def compute_oe(queries):
                pn_output_on_ood_data = self.log_p_y(sample['xs'], queries)['logits']
                # cross-entropy from softmax distribution to uniform distribution
                pn_output_on_ood_data_reshaped = pn_output_on_ood_data.view(-1, pn_output_on_ood_data.size(2))
                outlier_exposure_loss = -(pn_output_on_ood_data_reshaped.mean(1) - torch.logsumexp(pn_output_on_ood_data_reshaped, dim=1)).mean()
                return outlier_exposure_loss

            outlier_exposure_loss = compute_oe(sample['ooc_xq'])
            loss_val = loss_val + self.oe_lambda * outlier_exposure_loss
            ret_state['oe'] = outlier_exposure_loss.item()


        ret_state['loss'] = loss_val.item()
        return loss_val, ret_state


    def compute_embeddings(self, xs, xq):
        _log_p_y = self.log_p_y(xs, xq)
        zs, zq = _log_p_y['supports'], _log_p_y['zq']
        return zs, zq, _log_p_y




f_softplus = nn.functional.softplus
torch_pi = torch.from_numpy(np.ones(1)*np.pi)

def f_logp(mean, std , x):
    """ returns (N, ) logp according to a diagonal Gaussian
    (verified to behave like torch.distributions.multivariate_normal.MultivariateNormal)
    """
    assert len(mean.shape) == 2 # (N, D)
    var = torch.pow(std, 2)
    k = x.shape[-1]
    return -.5 * torch.sum(torch.pow(x-mean, 2) / var, -1) -.5*torch.sum(torch.log(var), -1) -.5*k*torch.log(2*torch_pi.type_as(var))


class ABML(Protonet):
    def __init__(self, **kwargs):
        # MAGIC
        self.train_lr = 0.01
        self.approx = False
        self.n_infer_samples = kwargs.pop('n_infer_samples')

        self.n_way = kwargs.pop('n_way')
        self.n_support = kwargs.pop('n_shot')
        self.inner_kl_lambda = kwargs.pop('inner_kl_lambda')
        self.outer_kl_lambda = kwargs.pop('outer_kl_lambda')
        self.hyper_prior = torch.distributions.gamma.Gamma(
                            torch.tensor([kwargs.pop('gamma_a')]).cuda(),
                            torch.tensor([kwargs.pop('gamma_b')]).cuda())
        self.task_update_num = kwargs.pop('task_update_num')
        # Differences from original paper
        self.sample_q_without_sgd = kwargs.pop('sample_q_without_sgd')
        self.outer_loss_query_and_support = kwargs.pop('outer_loss_query_and_support')
        self.outer_loss_use_kl_phi = kwargs.pop('outer_loss_use_kl_phi')
        super(ABML, self).__init__(**kwargs)
        self.feat_dim = self.input_dim
        self.feature = self.encoder
        self.loss_fn = nn.CrossEntropyLoss()
        self.classifier = backbone.Linear_fw_bbb(self.feat_dim, self.n_way)
        self.classifier.bias.data.fill_(0)

    def _reparametrized_gaussian_sample(self, mean, std):
        return mean + torch.randn_like(std) * std

    def _log_gaussian_density(self, samples, mean, std):
        ipdb.set_trace()
        distr = torch.distributions.multivariate_normal.MultivariateNormal(mean, torch.diag(torch.pow(std, 2)))
        return distr.log_prob(samples)

    def forward(self,x,return_stats=False):
        scores  = self.classifier.forward(self.feature.forward(x))
        return scores

    def _get_weights(self):
        return list(self.feature.parameters())+list(self.classifier.parameters())

    def _flatten_weights(self, ws, fast=False):
        if fast:
            return torch.cat([w.fast.view(-1) for w in ws])
        else:
            return torch.cat([w.view(-1) for w in ws])

    def _inner_objective(self, x, y):
        # Fpass
        scores = self.forward(x, True)
        # Loss components
        nll = self.loss_fn( scores, y)

        ws = self._get_weights()
        if ws[0].fast is None:
            kl = 0
        else:
            sample0, mean0, std0 = self.feature.get_sample_stats()
            sample1, mean1, std1 = self.classifier.get_sample_stats()
            sample, mean, std = sample0+sample1, mean0+mean1, std0+std1
            sample = self._flatten_weights(sample, fast=True)
            mean = self._flatten_weights(mean)
            std = f_softplus(self._flatten_weights(std))
            kl = -f_logp(mean[None], std[None] , sample[None])[0] # Note we ignore the term qlogq in the KL
            # WARNING: this kl is numerically very large...
            kl /= sample.shape[0]

        return nll + self.inner_kl_lambda * kl

    def set_forward(self,xs):
        old_state = self.training
        self.train()
        x_a_i = xs.view(-1, *xs.size()[2:]) #support data
        y_a_i = torch.from_numpy(np.repeat(range(self.n_way), self.n_support)).cuda() #label for support data

        fast_parameters = self._get_weights() #the first gradient calcuated is based on original weight
        for weight in self._get_weights():
            weight.fast = None
        self.zero_grad()
        # Get the initial params
        # _, mean0, std0 = self.forward(x_a_i, True)
        for task_step in range(self.task_update_num):
            set_loss = self._inner_objective(x_a_i, y_a_i)
            grad = torch.autograd.grad(set_loss, fast_parameters, create_graph=True) #build full graph support gradient of gradient
            if self.approx:
                grad = [ g.detach()  for g in grad ] #do not calculate gradient of gradient if using first order approximation
            fast_parameters = []
            for k, weight in enumerate(self._get_weights()):
                #for usage of weight.fast, please see Linear_fw, Conv_fw in backbone.py
                if weight.fast is None:
                    weight.fast = weight - self.train_lr * grad[k] #create weight.fast
                else:
                    weight.fast = weight.fast - self.train_lr * grad[k] #create an updated weight.fast, note the '-' is not merely minus value, but to create a new weight.fast
                fast_parameters.append(weight.fast) #gradients calculated in line 45 are based on newest fast weight, but the graph will retain the link to old weight.fasts
        if not old_state:
            self.eval()

    def log_p_y(self, xs, xq, no_grad=False):
        """log_p_y

        Args:
            xs: support set (#way, #shot, ...)
            xq: support set (#query classes, #queries, ...)
            return_target_inds: set to True only if #way and #query index over the same classes
        Return:
            a dict of {
                'log_p_y': log p(y) (#query classes, #queries, #way)
                'target_inds': ...
                'logits': ...
            }
        """
        ret_dict = {}
        # xs.size() == (50, 3, 1, 28, 28)  for 50-way, 3-shot classification
        # xq.size() == (50, 5, 1, 28, 28)  for 50-way, 5 query points for each of the 50 classes
        n_class = xs.size(0)
        n_query_class = xq.size(0)
        n_support = xs.size(1)
        n_query = xq.size(1)

        assert not no_grad

        if not self.training:
            lpys = []
            if self.sample_q_without_sgd:
                self.set_forward(xs)

            for _ in range(self.n_infer_samples):
                if not self.sample_q_without_sgd:
                    self.set_forward(xs)
                x_b_i = xq.view(-1, *xs.size()[2:])  # Query data
                scores = self.forward(x_b_i).detach()
                logits = scores.view(n_query_class, n_query, -1)
                log_p_y = F.log_softmax(logits, dim=2)
                lpys.append(log_p_y)

            log_p_y = torch.logsumexp(torch.stack(lpys), 0) - torch.log(torch.ones(1) * len(lpys)).to(lpys[0].device)

        else:
            x_b_i = xq.view(-1, *xs.size()[2:])  # Query data
            self.set_forward(xs)
            scores = self.forward(x_b_i).detach()
            logits = scores.view(n_query_class, n_query, -1)
            log_p_y = F.log_softmax(logits, dim=2)


        ret_dict['log_p_y']  = log_p_y
        # Prepare target_inds
        if n_class == n_query_class:
            target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
            target_inds.requires_grad = False
            if xq.is_cuda:
                target_inds = target_inds.cuda()
            ret_dict['target_inds'] = target_inds

        return ret_dict

    def loss(self, sample):
        _log_p_y = self.log_p_y(sample['xs'], sample['xq'])
        log_p_y, target_inds= _log_p_y['log_p_y'], _log_p_y['target_inds']
        nll = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
        if self.outer_loss_query_and_support:
            _log_p_y = self.log_p_y(sample['xs'], sample['xs'])
            log_p_y, target_inds = _log_p_y['log_p_y'], _log_p_y['target_inds']
            nll = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
        # Prior
        msample0, mean0, std0 = self.feature.get_sample_stats()
        msample1, mean1, std1 = self.classifier.get_sample_stats()
        msample, mean, std = msample0+msample1, mean0+mean1, std0+std1
        msample = self._flatten_weights(msample, fast=True)
        mean = self._flatten_weights(mean, fast=True)
        std = f_softplus(self._flatten_weights(std, fast=True))
        kl = -f_logp(torch.zeros_like(msample[None]), torch.ones_like(msample[None]) , msample[None])[0]
        kl -= self.hyper_prior.log_prob(std).sum()
        kl /= msample.shape[0]

        loss_val = nll + self.outer_kl_lambda * kl

        _, y_hat = log_p_y.max(2)
        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()

        ret_state = {
            'acc': acc_val.item(),
            'class_loss': loss_val.item()
        }

        if self.method in ['outlier_exposure']:
            raise

        if self.ooe_lambda > 0:
            raise

        ret_state['loss'] = loss_val.item()
        return loss_val, ret_state

    def ooc_scores(self, sample):
        in_scores = {}
        out_scores = {}
        if self.f_name in ['spp', 'neg_ent', 'nn']:
            in_log_p_y = self.log_p_y(sample['xs'], sample['xq'])['log_p_y']
            out_log_p_y = self.log_p_y(sample['xs'], sample['ooc_xq'])['log_p_y']
            def _reshape_acq(x, f_acq):
                x = torch.reshape(x, [-1, x.shape[-1]])
                return f_acq(x)

            in_scores[self.f_name] = _reshape_acq(in_log_p_y, self.f_acq)
            out_scores[self.f_name] = _reshape_acq(out_log_p_y, self.f_acq)
        else:
            raise ValueError()
        return in_scores, out_scores


class MAML(Protonet):
    def __init__(self, **kwargs):
        self.n_way = kwargs.pop('n_way')
        self.n_support = kwargs.pop('n_shot')
        self.task_update_num = kwargs.pop('task_update_num')
        self.use_support_stats = kwargs.pop('use_support_stats')
        super(MAML, self).__init__(**kwargs)
        approx = False
        self.feature = self.encoder
        x = torch.zeros([2] + kwargs['x_dim'])
        self.feat_dim = self.encoder(x).view(2,-1).shape[-1]

        self.loss_fn = nn.CrossEntropyLoss(reduction='none')
        self.classifier = backbone.Linear_fw(self.feat_dim, self.n_way)
        self.classifier.bias.data.fill_(0)

        self.train_lr = 0.01
        self.approx = approx



    def forward(self,x):
        out  = self.feature.forward(x)
        scores  = self.classifier.forward(out)
        return scores

    def set_forward(self, xs, xq, mask, no_grad=False):
        if self.loss_fn.reduction != 'none':
            self.loss_fn = nn.CrossEntropyLoss(reduction='none')

        n_way, n_support = xs.shape[:2]
        x_a_i = xs.reshape(-1, *xs.size()[2:]) #support data
        x_b_i = xq.reshape(-1, *xs.size()[2:]) #query data
        y_a_i = torch.from_numpy(np.repeat(range(n_way), n_support)).to(xs.device) #label for support data
        # Normalize weights to be class-balanced
        mask = mask / mask.sum(1,keepdim=True)
        mask = mask.reshape(-1)
        # Filter out the non-used (this is important, otherwise the 'garbage' contributes to BN in the forward pass)
        hard_mask = mask > 0
        x_a_i = x_a_i[hard_mask]
        y_a_i = y_a_i[hard_mask]
        mask = mask[hard_mask]

        fast_parameters = list(self.feature.parameters())+list(self.classifier.parameters()) #the first gradient calcuated is based on original weight
        for weight in list(self.feature.parameters())+list(self.classifier.parameters()):
            weight.fast = None
        self.zero_grad()
        
        self.train()

        for task_step in range(self.task_update_num):
            scores = self.forward(x_a_i)
            losses = self.loss_fn( scores, y_a_i)

            set_loss = (losses * mask.reshape(-1)).sum() / mask.sum()
            grad = torch.autograd.grad(set_loss, fast_parameters, create_graph=not no_grad) #build full graph support gradient of gradient
            if self.approx or no_grad:
                grad = [ g.detach()  for g in grad ] #do not calculate gradient of gradient if using first order approximation
            fast_parameters = []
            for k, weight in enumerate(list(self.feature.parameters())+list(self.classifier.parameters())):
                #for usage of weight.fast, please see Linear_fw, Conv_fw in backbone.py
                if weight.fast is None:
                    weight.fast = weight - self.train_lr * grad[k] #create weight.fast
                else:
                    weight.fast = weight.fast - self.train_lr * grad[k] #create an updated weight.fast, note the '-' is not merely minus value, but to create a new weight.fast
                fast_parameters.append(weight.fast) #gradients calculated in line 45 are based on newest fast weight, but the graph will retain the link to old weight.fasts

        if self.use_support_stats:
            self.eval()
        scores = self.forward(x_b_i)

        return scores

    def log_p_y(self, xs, xq, no_grad=False, mask=None):
        """log_p_y

        Args:
            xs: support set (#way, #shot, ...)
            xq: support set (#query classes, #queries, ...)
            return_target_inds: set to True only if #way and #query index over the same classes
        Return:
            a dict of {
                'log_p_y': log p(y) (#query classes, #queries, #way)
                'target_inds': ...
                'logits': ...
            }
        """

        ret_dict = {}
        # xs.size() == (50, 3, 1, 28, 28)  for 50-way, 3-shot classification
        # xq.size() == (50, 5, 1, 28, 28)  for 50-way, 5 query points for each of the 50 classes
        n_class = xs.size(0)
        n_query_class = xq.size(0)
        n_support = xs.size(1)
        n_query = xq.size(1)

        if  mask is None:
            mask = torch.ones(n_class, n_support).to(xs.device)
        scores = self.set_forward(xs, xq, mask, no_grad)
        if no_grad:
            scores = scores.detach()

        x = torch.cat([xs.reshape(n_class * n_support, *xs.size()[2:]),
                       xq.reshape(n_query_class * n_query, *xq.size()[2:])], 0)


        if self.use_support_stats:
            self.eval()
        z = self.encoder(x, no_grad=no_grad)
        if no_grad:
            z = z.detach()

        z_dim = z.size(-1)
        supports = z[:n_class*n_support].view(n_class, n_support, z_dim)
        ret_dict['supports'] = supports
        ret_dict['zq'] = z[n_class*n_support:]

        logits = scores.view(n_query_class, n_query, -1)
        log_p_y = F.log_softmax(logits, dim=2)
        ret_dict['logits'] = logits
        ret_dict['log_p_y']  = log_p_y
        # Prepare target_inds
        if n_class == n_query_class:
            target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
            target_inds.requires_grad = False
            if xq.is_cuda:
                target_inds = target_inds.cuda()
            ret_dict['target_inds'] = target_inds
        return ret_dict

    def loss(self, sample):
        _log_p_y = self.log_p_y(sample['xs'], sample['xq'])
        log_p_y, target_inds, _logits= _log_p_y['log_p_y'], _log_p_y['target_inds'], _log_p_y['logits']

        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
        _, y_hat = log_p_y.max(2)
        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()

        ret_state = {
            'acc': acc_val.item(),
            'class_loss': loss_val.item()
        }

        if self.method in ['outlier_exposure']:
            raise

        if self.ooe_lambda > 0:
            raise

        ret_state['loss'] = loss_val.item()
        return loss_val, ret_state

    def compute_embeddings(self, xs, xq, no_grad=False):
        _log_p_y = self.log_p_y(xs, xq, no_grad=no_grad)
        zs, zq = _log_p_y['supports'], _log_p_y['zq']
        return zs, zq, _log_p_y


    def backbone_fpass(self, xs, xq, repeat_query=False,no_grad=False):
        zs, zq, _log_p_y = self.compute_embeddings(xs, xq, no_grad=no_grad)
        lcbo_inp = prepare_lcbo_input(zs, zq, repeat_query=repeat_query)  # Prepare OEC input
        dists = -_log_p_y['logits'].view(-1, xs.size(0))  # ED
        return lcbo_inp, -torch.min(dists, -1)[0]

    def ooc_scores(self, sample):
        in_scores = {}
        out_scores = {}

        if self.f_name == 'spp':
            in_log_p_y = self.log_p_y(sample['xs'], sample['xq'], no_grad=True)['log_p_y']
            out_log_p_y = self.log_p_y(sample['xs'], sample['ooc_xq'], no_grad=True)['log_p_y']

            def _reshape_acq(x, f_acq):
                x = torch.reshape(x, [-1, x.shape[-1]])
                return f_acq(x)

            in_scores[self.f_name] = _reshape_acq(in_log_p_y, self.f_acq).detach()
            out_scores[self.f_name] = _reshape_acq(out_log_p_y, self.f_acq).detach()

        elif self.f_name == 'ed':
            def compute_closest_dist(queries):
                zs, zq, _ = self.compute_embeddings(sample['xs'], queries)
                n_way, n_shot, dim = zs.shape
                # zq (n, dim) , zs (way, shot, dim)
                proto = zs.mean(1)  # (way, dim)
                dists = torch.pow(zq[:,None] - proto[None], 2).sum(-1)  # (n, way)
                return -dists.min(1)[0]

            in_scores[self.f_name] = compute_closest_dist(sample['xq'])
            out_scores[self.f_name] = compute_closest_dist(sample['ooc_xq'])

        elif self.f_name == 'oec':  # Compute the OEC in-dist and OOD scores
            all_inp, _ = self.backbone_fpass(sample['xs'], torch.cat([sample['xq'],sample['ooc_xq']],1), repeat_query=True, no_grad=True)
            lcbo_output = self.lcbo(all_inp.view(-1, all_inp.shape[-1]))
            lcbo_output = lcbo_output.view(n_way, 2*n_way*n_query)
            in_lcbo_output = lcbo_output[:, :n_way*n_query]
            out_lcbo_output = lcbo_output[:, n_way*n_query:]

            
            # Aggregation
            if self.lcbo_aggregation == 'max':
                in_scores[self.f_name] = in_lcbo_output.max(0)[0].squeeze().detach().cpu()
                out_scores[self.f_name] = out_lcbo_output.max(0)[0].squeeze().detach().cpu()
            else:
                raise 

        return in_scores, out_scores


def create_model(**kwargs):
    model_class = {
        'protonet': Protonet,
        'maml': MAML,
        'abml': ABML,
    }
    ARCHS = { 
          'conv4':    partial(backbone.Conv4, imgSize=kwargs['data.x_dim'][1], idim=kwargs['data.x_dim'][0]),
          'conv4bbb': backbone.Conv4BBB,
          'resnet10_32': partial(backbone.ResNet10_32, nc=kwargs['data.x_dim'][0]),
          'resnet10_84': partial(backbone.ResNet10_84, nc=kwargs['data.x_dim'][0]),
        }

    if kwargs['model.class'] == 'maml':
        backbone.ConvBlock.maml = True
        backbone.SimpleBlock.maml = True
        backbone.ResNet.maml = True

    if kwargs['model.encoder'] == 'resnet10':
        arch = 'resnet10_32' if kwargs['data.dataset'] == 'cifar100' else 'resnet10_84'
    else:
        arch = kwargs['model.encoder']
    encoder = ARCHS[arch]()


    if kwargs['model.class'] == 'abml':
        assert kwargs['model.encoder'] == 'conv4bbb'
        assert kwargs['data.way'] == kwargs['data.test_way'] and kwargs['data.shot'] == kwargs['data.test_shot']
        
        
        return model_class[kwargs['model.class']](
                        sample_q_without_sgd=kwargs['abml.sample_q_without_sgd'],
                        outer_loss_query_and_support=kwargs['abml.outer_loss_query_and_support'],
                        outer_loss_use_kl_phi=kwargs['abml.outer_loss_use_kl_phi'],
                        task_update_num=kwargs['maml.task_update_num'],
                        n_way=kwargs['data.way'],
                        n_shot=kwargs['data.shot'],
                        n_infer_samples=kwargs['abml.n_infer_samples'],
                        inner_kl_lambda=kwargs['abml.inner_kl_lambda'],
                        outer_kl_lambda=kwargs['abml.outer_kl_lambda'],
                        gamma_a=kwargs['abml.gamma_a'],
                        gamma_b=kwargs['abml.gamma_b'],
                        encoder=encoder,
                        f_name=kwargs['model.f_acq'],
                        x_dim=kwargs['data.x_dim'],
                        method=kwargs['model.method'],
                        oe_lambda=kwargs['model.oe_lambda'],  # outlier exposure contribution to the loss
                        ooe_lambda=kwargs['model.ooe_lambda'],
                        lcbo_arch=kwargs['model.lcbo_arch'],
                        lcbo_aggregation=kwargs['model.lcbo_aggregation'])
    elif kwargs['model.class'] == 'maml':
        assert kwargs['data.way'] == kwargs['data.test_way'] and kwargs['data.shot'] == kwargs['data.test_shot']
        return model_class[kwargs['model.class']](
                        task_update_num=kwargs['maml.task_update_num'],
                        n_way=kwargs['data.way'],
                        n_shot=kwargs['data.shot'],
                        encoder=encoder,
                        f_name=kwargs['model.f_acq'],
                        x_dim=kwargs['data.x_dim'],
                        method=kwargs['model.method'],
                        oe_lambda=kwargs['model.oe_lambda'],  # outlier exposure contribution to the loss
                        ooe_lambda=kwargs['model.ooe_lambda'],
                        lcbo_arch=kwargs['model.lcbo_arch'],
                        lcbo_aggregation=kwargs['model.lcbo_aggregation'],
                        use_support_stats=kwargs['model.use_support_stats'])
    else:
        return model_class[kwargs['model.class']](encoder=encoder,
                        f_name=kwargs['model.f_acq'],
                        x_dim=kwargs['data.x_dim'],
                        method=kwargs['model.method'],
                        oe_lambda=kwargs['model.oe_lambda'],  # outlier exposure contribution to the loss
                        ooe_lambda=kwargs['model.ooe_lambda'],
                        lcbo_arch=kwargs['model.lcbo_arch'],
                        lcbo_aggregation=kwargs['model.lcbo_aggregation'])
