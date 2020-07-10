"""Evaluate few-shot active learning and few-shot semi-supervised learning.
"""
import os
import sys
import ipdb
import json
import pickle
import random
import argparse
from tqdm import tqdm
from collections import defaultdict, OrderedDict, namedtuple

import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

import torchnet as tnt
from torchvision import transforms

# Local imports
import logger
from utils import filter_opt, compute_ece, prep_accs
from protonet import Protonet
from anomaly import load_ood_data , show_ood_detection_results_softmax

from csv_logger import CSVLogger, plot_csv
from torch.optim.lr_scheduler import StepLR

from classify_classic import ResNetClassifier
from data import get_dataset, get_transform, load_episode, cifar_normalize
from confidence.base import BaseConfidence, FSCConfidence
from confidence.dm import DeepMahala, DMConfidence
from confidence.oec import OECConfidence, OEC
from confidence.deep_oec import DeepOECConfidence
from create_ensemble import Ensemble
from eval_ood import eval_ood_aurocs, score_batch
import wandb
from copy import deepcopy
import sys
import aggregate_eval_active

parser = argparse.ArgumentParser(description='Evaluate few-shot active learning and few-shot semi-supervised learning')

default_model_path = 'results/best_model.pt'
default_output_dir = ''  # In this case, the directory of model.model_path will be used.
parser.add_argument('--model.model_path', type=str, default=default_model_path, metavar='MODELPATH',
                    help="location of pretrained model to evaluate (default: {:s})".format(default_model_path))
parser.add_argument('--data.test_way', type=int, default=0, metavar='TESTWAY',
                    help="number of classes per episode in test. 0 means same as model's data.test_way (default: 0)")
parser.add_argument('--data.test_shot', type=int, default=0, metavar='TESTSHOT',
                    help="number of support examples per class in test. 0 means same as model's data.shot (default: 0)")
parser.add_argument('--data.test_query', type=int, default=0, metavar='TESTQUERY',
                    help="number of query examples per class in test. 0 means same as model's data.test_query (default: 0)")
parser.add_argument('--data.test_episodes', type=int, default=1000, metavar='NTEST',
                    help="number of test episodes per epoch (default: 1000)")
parser.add_argument('--model.f_acq', type=str, default='neg_ent', metavar='ACQ',
                    help="{'spp', 'neg_ent', 'nn', 'ooc_support'}")
parser.add_argument('--use_residual', action='store_true', default=False,
                    help='If True, input the variance of the support examples to the OEC in addition to the mean')
parser.add_argument('--arch', type=str, default='100,100',
                    help="OEC hidden layers")
parser.add_argument('--model.decision', type=str, default='baseline',
                    help="see protonet.py")
parser.add_argument('--prefix', type=str, default= '')
parser.add_argument('--abml.n_infer_samples', type=int, default=2)
parser.add_argument('--maml.task_update_num', type=int, default=5)
parser.add_argument('--output_dir', type=str, default=default_output_dir, metavar='EVALOUTPUTDIR',
                    help="location to write the evaluation trace and pkl (defaults to the parent dir of model.model_path).")
parser.add_argument('--budget_active', type=int, default=0)
parser.add_argument('--soft_ssl', type=int, default=0)
parser.add_argument('--max_temp_select_iter', type=int, default=10)
parser.add_argument('--exp_name', type=str, required=True)
parser.add_argument('--dataset', required=True, type=str, choices=['cifar-fs', 'miniimagenet'])
parser.add_argument('--dataroot', required=True, type=str)
parser.add_argument('--db', type=int, default=0)
parser.add_argument('--ood_method', type=str, required=True)
parser.add_argument('--oec_path', type=str, default='-')
parser.add_argument('--n_unlabeled_per_class', type=int, default=1)
parser.add_argument('--n_distractor_per_class', type=int, default=1)

def compute_acc(model, xs, xq,  mask):
    lpy_dic = model.log_p_y(xs, xq, mask=mask, no_grad=True)
    log_p_y, target_inds = lpy_dic['log_p_y'], lpy_dic['target_inds']
    conf, y_hat = log_p_y.max(-1)
    return torch.eq(y_hat, target_inds.squeeze()).float().view(-1).detach()


def main(opt):


    eval_exp_name = opt['exp_name']
    device = 'cuda:0' 

    # Load data
    if opt['dataset'] == 'cifar-fs':
        train_data = get_dataset('cifar-fs-train-train', opt['dataroot'])
        val_data = get_dataset('cifar-fs-val', opt['dataroot'])
        test_data = get_dataset('cifar-fs-test', opt['dataroot'])
        tr = get_transform('cifar_resize_normalize')
        normalize = cifar_normalize
    elif  opt['dataset'] == 'miniimagenet':
        train_data = get_dataset('miniimagenet-train-train', opt['dataroot'])
        val_data = get_dataset('miniimagenet-val', opt['dataroot'])
        test_data = get_dataset('miniimagenet-test', opt['dataroot'])
        tr = get_transform('cifar_resize_normalize_84')
        normalize = cifar_normalize


    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)

    
    if opt['db']:
        ood_distributions = ['ooe', 'gaussian']
    else:
        ood_distributions = ['ooe', 'gaussian', 'svhn']
        # ood_distributions = ['ooe', 'gaussian', 'rademacher', 'texture3', 'svhn','tinyimagenet','lsun']

    ood_tensors = [('ooe', None)] + [(out_name, load_ood_data({
                          'name': out_name,
                          'ood_scale': 1,
                          'n_anom': 10000,
                        })) for out_name in ood_distributions[1:]]

    # Load trained model
    loaded = torch.load(opt['model.model_path'])
    if not isinstance(loaded, OrderedDict):
        protonet = loaded
    else:
        classifier = ResNetClassifier(64, train_data['im_size']).to(device)
        classifier.load_state_dict(loaded)
        protonet = Protonet(classifier.encoder)
    encoder = protonet.encoder
    encoder.eval()
    encoder.to(device)
    protonet.eval()
    protonet.to(device)

    # Init Confidence model
    if opt['ood_method'] == 'deep-ed-iso':
        deep_mahala_obj = DeepMahala(None, None, None, encoder, device,num_feats=encoder.depth, num_classes=train_data['n_classes'], pretrained_path="", fit=False, normalize=None)
        conf = DMConfidence(deep_mahala_obj, {'ls':range(encoder.depth),'reduction':'max','g_magnitude':0}, True, 'iso').to(device)
    elif opt['ood_method'] == 'native-spp':
        conf = FSCConfidence(protonet, 'spp')
    elif opt['ood_method'] == 'oec':
        oec_opt = json.load(
                open(os.path.join(os.path.dirname(opt['oec_path']), 'args.json'), 'r')
            )
        init_sample = load_episode(train_data, tr, oec_opt['data.test_way'], oec_opt['data.test_shot'], oec_opt['data.test_query'], device)
        if oec_opt['confidence_method'] == 'oec':
          oec_conf = OECConfidence(None, protonet, init_sample, oec_opt)
        else:
          oec_conf = DeepOECConfidence(None, protonet, init_sample, oec_opt)
        oec_conf.load_state_dict(
              torch.load(opt['oec_path'])
            )
        oec_conf.eval()
        oec_conf.to(device)
        conf = oec_conf
      

    # Turn confidence score into a threshold based classifier
    # Select threshold by "max-accuracy"
    # Select temperature by "best-calibration" in the binary problem
    # done using the meta-train set
    in_scores = []
    out_scores = []
    for n in tqdm(range(100)):
        sample = load_episode(train_data, tr, opt['data.test_way'], opt['data.test_shot'], opt['data.test_query'], device)
        in_score, out_score = score_batch(conf, sample)
        in_scores.append(in_score)
        out_scores.append(out_score)
    in_scores = torch.cat(in_scores)
    out_scores = torch.cat(out_scores)

    def _compute_acc(in_scores, out_scores, t):
        N = len(in_scores ) + len(out_scores)
        return (torch.sum(in_scores >= t) + torch.sum(out_scores < t)).item() / float(N)

    best_threshold = torch.min(in_scores)
    best_acc = _compute_acc(in_scores, out_scores, best_threshold)

    for t in in_scores:
        acc = _compute_acc(in_scores, out_scores, t)
        if acc > best_acc:
            best_acc = acc
            best_threshold = t

    def _compute_confs(in_scores, out_scores, t, temp):
        in_p = torch.sigmoid( (in_scores - t) / temp )
        corrects = in_p >= .5
        confs = torch.max(torch.stack([in_p, 1-in_p]), 0)[0]
        out_p = torch.sigmoid( (out_scores - t) / temp )
        corrects = torch.cat([corrects, out_p < .5])
        confs = torch.cat([confs, torch.max(torch.stack([out_p, 1-out_p]),0)[0]])
        return confs, corrects

    def compute_eces(candidate_temps, in_scores, out_scores, best_threshold):
        eces = []
        for temp in candidate_temps:
            confs, corrects = _compute_confs(in_scores, out_scores, best_threshold, temp)
            ece = compute_ece(*prep_accs(confs.numpy(), corrects.numpy(), bins=20))
            eces.append(ece)
        return eces


    min_log_temp = -1
    log_interval = 2
    npts = 10

    for _ in range(opt['max_temp_select_iter']):
        print("..selecting temperature")
        candidate_temps = np.logspace(min_log_temp,min_log_temp+log_interval,npts)
        eces = compute_eces(candidate_temps, in_scores, out_scores, best_threshold)
        min_idx = np.argmin(eces)
        if min_idx == 0:
            min_log_temp -= log_interval // 2
        elif min_idx == npts -  1:
            min_log_temp += log_interval // 2
        else:
            break

    best_ece = eces[min_idx]
    best_temp = candidate_temps[min_idx]

    print(f"Best ACC:{best_acc}, thresh:{best_threshold}, Best ECE:{best_ece}, temp:{best_temp}")



    def get_95_percent_ci(std):
      """Computes the 95% confidence interval from the standard deviation."""
      return std * 1.96 / np.sqrt(data_opt['data.test_episodes'])

    active_supervised = defaultdict(list)
    active_augmented = defaultdict(list)
    ssl_soft = defaultdict(list)
    ssl_hard = defaultdict(list)
    # for ood_idx, curr_ood in tqdm(enumerate(all_distributions)):
    for curr_ood, ood_tensor in ood_tensors:
        
        in_scores = defaultdict(list)
        out_scores = defaultdict(list)
        # Compute and collect scores for all examples
        aurocs, auprs, fprs = defaultdict(list),defaultdict(list),defaultdict(list)

        for n in tqdm(range(opt['data.test_episodes'])):
            n_total_query = np.max([opt['data.test_query'] + opt['n_unlabeled_per_class'],
                                    opt['n_distractor_per_class']])
            sample = load_episode(test_data, tr, opt['data.test_way'], opt['data.test_shot'], n_total_query, device)
            if curr_ood !='ooe':
                  bs = opt['data.test_way']*opt['data.test_query']
                  ridx =  np.random.permutation(bs)
                  sample['ooc_xq'] = torch.stack([tr(x) for x  in ood_tensor[ridx]]).to(device)
                  way, _, c, h, w = sample['xq'].shape
                  sample['ooc_xq'] = sample['ooc_xq'].reshape(way, -1, c,h,w)
                  # if curr_ood in ['gaussian', 'rademacher']:
                  #   sample['ooc_xq'] *= 4
                
            all_xq = sample['xq'].clone()
            sample['xq'] = all_xq[:,:opt['n_unlabeled_per_class']]          # Unlabelled pool
            sample['xq2'] = all_xq[:,opt['n_unlabeled_per_class']:opt['n_unlabeled_per_class']+opt['data.test_query']]         # Final test queries
            sample['ooc_xq'] = sample['ooc_xq'][:,:opt['n_distractor_per_class']]



            """
            1.  OOD classification on the 'unlabelled' set
            """
            # In vs Out
            in_score, out_score = score_batch(conf, sample)

            num_in = in_score.shape[0]
            confs, corrects = _compute_confs(in_score, out_score, best_threshold, best_temp)
            in_mask = corrects[:num_in].reshape(sample['xq'].size(0), sample['xq'].size(1)).float().to(device)
            out_mask = 1 - corrects[num_in:].reshape(sample['ooc_xq'].size(0), sample['ooc_xq'].size(1)).float().to(device)


            """
            2.0
            """
            budget_active = in_score.size(0)
            scores = torch.cat([in_score, out_score], -1)
            ipdb.set_trace()
            selected_inds = torch.sort(scores)[1][scores.size(0)-budget_active:]
            selected_inds_in = selected_inds[selected_inds<in_score.size(0)]
            budget_mask = torch.zeros(in_score.size(0)).to(device)
            budget_mask.scatter_(0, selected_inds_in.to(device).long(), 1)
            budget_mask = budget_mask.reshape(sample['xq'].size(0), sample['xq'].size(1)).float().to(device)


            """
            2.  Add labels to the predicted unlabelled examples
            """
            # Collect the incorrectly kept OOD examples
            included_distractors = sample['ooc_xq'][out_mask.byte()]
            # Pad them to N-way multiples, and assign random labels (done simply by reshaping)
            n_way = sample['xs'].shape[0]
            im_shape = list(sample['xs'].shape[2:])
            n_res = n_way - (included_distractors.shape[0] % n_way)
            distractor_mask = torch.ones([included_distractors.shape[0]]).to(device)

            zeros = torch.zeros([n_res]+im_shape).to(device)
            included_distractors = torch.cat([included_distractors,zeros])
            distractor_mask = torch.cat([distractor_mask, torch.zeros([n_res]).to(device)])
            # the reason we permute is to spread the padded zero across ways
            included_distractors = included_distractors.reshape([ -1, n_way]+im_shape).permute(1,0,2,3,4)
            distractor_mask = distractor_mask.reshape([-1, n_way]).permute(1,0)


            """
            2.5 SSL
            """
            # predict k-way using classifier
            n_way, n_aug_shot, n_ch, n_dim, _ = sample['xq'].shape
            lpy_dic = protonet.log_p_y(sample['xs'], sample['xq'], mask=None)
            log_p_y, target_inds = lpy_dic['log_p_y'], lpy_dic['target_inds']

            preds = log_p_y.max(-1)[1]

            def reorder(unlabelled, preds, py, make_soft=True):
                if py is not None:
                    reshaped_py = py.reshape(-1)
                n_way, n_aug_shot, n_ch, n_dim, _ = unlabelled.shape
                reshaped_unlabelled = unlabelled.reshape(n_aug_shot*n_way, n_ch, n_dim, n_dim)
                reshaped_predicted_labels = preds.reshape(-1)
                unlabelled = torch.zeros((n_way, n_aug_shot*n_way, n_ch, n_dim, n_dim))
                mask = torch.zeros((n_way, n_aug_shot*n_way))
                for idx, label in enumerate(reshaped_predicted_labels):
                    unlabelled[label, idx] = reshaped_unlabelled[idx] # (n_shot, ...)
                    if make_soft:
                        mask[label, idx] = reshaped_py[idx]
                    else:
                        mask[label, idx] = 1 # (n_shot, )
                return unlabelled.to(device), mask.to(device)
            gt_in_unlabelled, gt_in_weights = reorder(sample['xq'], preds, log_p_y.max(-1)[0].exp(), True)
            _, in_mask_reordered = reorder(sample['xq'], preds, in_mask, True)

            # for the gt OOD ones
            lpy_dic = protonet.log_p_y(sample['xs'], sample['ooc_xq'], mask=None)
            log_p_y= lpy_dic['log_p_y']

            preds = log_p_y.max(-1)[1]
            gt_ood_unlabelled, gt_ood_weights = reorder(sample['ooc_xq'], preds, log_p_y.max(-1)[0].exp(), True)
            _,  out_mask_reordered= reorder(sample['ooc_xq'], preds, out_mask, True)


            # Support + ALL unlabelled
            _ssl_soft = compute_acc(protonet,
                                    torch.cat([sample['xs'],gt_in_unlabelled ,gt_ood_unlabelled],1),
                                    sample['xq2'],
                                    torch.cat([torch.ones(sample['xs'].shape[:2]).to(device),  gt_in_weights, gt_ood_weights],1)
                                   )
            _acc_hard = compute_acc(protonet,
                                    torch.cat([sample['xs'],gt_in_unlabelled ,gt_ood_unlabelled],1),
                                    sample['xq2'],
                                    torch.cat([torch.ones(sample['xs'].shape[:2]).to(device), in_mask_reordered* gt_in_weights, out_mask_reordered*gt_ood_weights],1)
                                   )


            """
            3.  Evaluate k-way accuracy after adding examples
            """
            _active_supervised = compute_acc(protonet,
                                             sample['xs'],
                                             sample['xq2'],
                                             None
                                            )


            # Support + Budgeted unlabelled
            _active_augmented = compute_acc(protonet,
                                            torch.cat([sample['xs'], sample['xq']],1),
                                            sample['xq2'],
                                            torch.cat([torch.ones(sample['xs'].shape[:2]).to(device), budget_mask],1)
                                           )

            ssl_soft[curr_ood].append(_ssl_soft)
            ssl_hard[curr_ood].append(_acc_hard)
            active_supervised[curr_ood].append(_active_supervised)
            active_augmented[curr_ood].append(_active_augmented)

    if not os.path.exists(opt['output_dir']):
        os.makedirs(opt['output_dir'])

    pickle.dump(
            (ssl_soft,
            ssl_hard,
            active_supervised,
            active_augmented), open(os.path.join(opt['output_dir'], f'eval_active_{eval_exp_name}.pkl'), 'wb')
        )

    print("===> Aggregating results")
    aggr_args  = namedtuple('Arg', ('exp_dir','f_acq'))(
                exp_dir=opt['output_dir'], 
                f_acq='conv4')
    aggregate_eval_active.main(aggr_args)
    print('===> Done')
    sys.exit()

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    args = vars(parser.parse_args())
    main(args)
