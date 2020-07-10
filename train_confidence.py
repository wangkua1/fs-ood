import os
import sys
import ipdb
import json
import pickle
import random
import argparse
from tqdm import tqdm
from collections import defaultdict, OrderedDict

import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

import torchnet as tnt
from torchvision import transforms

# Local imports
import logger
from utils import filter_opt
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
from eval_ood import eval_ood_aurocs
import wandb
from copy import deepcopy

device = 'cuda:0' 

parser = argparse.ArgumentParser(description='Train OEC network')

default_model_path = 'results/best_model.pt'
default_eval_output_dir = ''
parser.add_argument('--model.model_path', type=str, default=default_model_path, metavar='MODELPATH',
                    help='location of pretrained model to evaluate (default: {:s})'.format(default_model_path))
parser.add_argument('--data.test_way', type=int, default=0, metavar='TESTWAY',
                    help='number of classes per episode in test. 0 means same as data.test_way (default: 0)')
parser.add_argument('--data.test_shot', type=int, default=0, metavar='TESTSHOT',
                    help='number of support examples per class in test. 0 means same as data.shot (default: 0)')
parser.add_argument('--data.test_query', type=int, default=0, metavar='TESTQUERY',
                    help='number of query examples per class in test. 0 means same as data.test_query (default: 0)')
parser.add_argument('--data.test_episodes', type=int, default=1000)
parser.add_argument('--lr', type=float, default=1e-3,
                    help='Learning rate')
parser.add_argument('--wd', type=float, default=0,
                    help='Weight decay')
parser.add_argument('--eval_every_outer', type=int, default=100,
                    help='Evaluate after this many outer iterations')
parser.add_argument('--exp_name', type=str, default='acq',
                    help='prefix of trace file, placed in the same dir as ckpt')
parser.add_argument('--arch', type=str, default='500,500',
                    help='OEC hidden layers')
#
parser.add_argument('--output_dir', type=str, default=default_eval_output_dir, metavar='EVALOUTPUTDIR',
                    help=".")
parser.add_argument('--train_iter', type=int, default=100, help="Num updates to OEC")
parser.add_argument('--residual', type=int, default=0)
parser.add_argument('--lrsche_step_size', type=int, default=100000000)
parser.add_argument('--lrsche_gamma', type=float, default=0.1)
parser.add_argument('--oec_track_running_stats', type=int, default=1)
## 
parser.add_argument('--confidence_method', type=str, required=True, choices=['oec','deep-oec','dm-iso'])
parser.add_argument('--interpolate', type=int, default=0)
parser.add_argument('--full_supervision', type=int, default=0)
parser.add_argument('--eval_in_train', type=int, default=0)
parser.add_argument('--in_out_1_batch', type=int, default=0)
parser.add_argument('--oec_norm_type', type=str, default='bn', choices=['bn','in','gn'])
parser.add_argument('--oec_depth', type=str, default='all', choices=['penult','all'])
parser.add_argument('--oec_depth_aggregation', type=str, default='input', choices=['input','output-sum','output-lse'])
parser.add_argument('--resume', type=int, default=0)
parser.add_argument('--db', type=int, default=0)
parser.add_argument('--input_regularization', type=str, default='-', choices=['-', 'oe'])
parser.add_argument('--input_regularization_percent', type=float, default=0.5)
parser.add_argument('--oec_embed_in_eval', type=int, default=0)
parser.add_argument('--dataset', required=True, type=str, choices=['cifar-fs', 'miniimagenet'])
parser.add_argument('--dataroot', required=True, type=str)
parser.add_argument('--pretrained_oec_path', type=str, default='')
parser.add_argument('--n_ensemble', type=int, default=5)
parser.add_argument('--ooe_only', type=int, default=0)

    
def _print_and_log(print_str, trace_file):
    print(print_str)
    with open(trace_file, 'a') as f:
        json.dump(print_str, f)
        f.write('\n')

def compute_loss_bce(in_scores, out_scores, mean_center=True):
    logits = torch.cat([in_scores, out_scores], 0)  # (1000)
    if mean_center:
        logits = (logits - logits.mean())#/(logits.var() + 1e-5)
    target = torch.cat([torch.ones(len(in_scores)), torch.zeros(len(out_scores))],0).type_as(logits)
    ooe_loss = F.binary_cross_entropy_with_logits(logits, target)
    acc = (logits > 0).float().eq(target).float().mean()
    return ooe_loss, acc



def main(opt):

    # Logging
    trace_file = os.path.join(opt['output_dir'], '{}_trace.txt'.format(opt['exp_name']))
    
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

    if opt['input_regularization'] == 'oe':
        reg_data = load_ood_data({
                          'name': 'tinyimages',
                          'ood_scale': 1,
                          'n_anom': 50000,
                        })

    if not opt['ooe_only']:
        if opt['db']:
            ood_distributions = ['ooe', 'gaussian']
        else:
            ood_distributions = ['ooe', 'gaussian', 'rademacher', 'texture3', 'svhn','tinyimagenet','lsun']
            if opt['input_regularization'] == 'oe':
                ood_distributions.append('tinyimages')

        ood_tensors = [('ooe', None)] + [(out_name, load_ood_data({
                              'name': out_name,
                              'ood_scale': 1,
                              'n_anom': 10000,
                            })) for out_name in ood_distributions[1:]]

    # Load trained model
    loaded = torch.load(opt['model.model_path'])
    if not isinstance(loaded, OrderedDict):
        fs_model = loaded
    else:
        classifier = ResNetClassifier(64, train_data['im_size']).to(device)
        classifier.load_state_dict(loaded)
        fs_model = Protonet(classifier.encoder)
    fs_model.eval()
    fs_model = fs_model.to(device)


    # Init Confidence Methods
    if opt['confidence_method'] == 'oec':
        init_sample = load_episode(train_data, tr, opt['data.test_way'], opt['data.test_shot'], opt['data.test_query'], device)
        conf_model = OECConfidence(None, fs_model, init_sample, opt)
    elif opt['confidence_method'] == 'deep-oec':
        init_sample = load_episode(train_data, tr, opt['data.test_way'], opt['data.test_shot'], opt['data.test_query'], device)
        conf_model = DeepOECConfidence(None, fs_model, init_sample, opt)
    elif opt['confidence_method'] == 'dm-iso':
        encoder = fs_model.encoder
        deep_mahala_obj = DeepMahala(None, None, None, encoder, device,num_feats=encoder.depth, num_classes=train_data['n_classes'], pretrained_path="", fit=False, normalize=None)

        conf_model = DMConfidence(deep_mahala_obj, {'ls':range(encoder.depth),'reduction':'max','g_magnitude':.1}, True, 'iso')

    if opt['pretrained_oec_path']:
        conf_model.load_state_dict(
              torch.load(opt['pretrained_oec_path'])
            )
        

    conf_model.to(device)
    print(conf_model)

    optimizer = optim.Adam(conf_model.confidence_parameters(), lr=opt['lr'], weight_decay=opt['wd'])
    scheduler = StepLR(optimizer, step_size=opt['lrsche_step_size'], gamma=opt['lrsche_gamma'])

    num_param = sum(p.numel() for p in conf_model.confidence_parameters())
    print(f"Learning Confidence, Number of Parameters -- {num_param}")

    if conf_model.pretrain_parameters() is not None:
        pretrain_optimizer = optim.Adam(conf_model.pretrain_parameters(), lr=10)
        pretrain_iter = 100

    start_idx = 0
    if opt['resume']:
        last_ckpt_path = os.path.join(opt['output_dir'], 'last_ckpt.pt')
        if os.path.exists(last_ckpt_path):
            try:
                last_ckpt = torch.load(last_ckpt_path)
                if 'conf_model' in last_ckpt:
                    conf_model = last_ckpt['conf_model'] 
                else:
                    sd = last_ckpt['conf_model_sd']
                    conf_model.load_state_dict(sd)
                optimizer = last_ckpt['optimizer'] 
                pretrain_optimizer = last_ckpt['pretrain_optimizer'] 
                scheduler = last_ckpt['scheduler'] 
                start_idx = last_ckpt['outer_idx'] 
                conf_model.to(device)
            except EOFError:
                print("\n\nResuming but got EOF error, starting from init..\n\n")
    

    wandb.run.name = opt['exp_name']
    wandb.run.save()
    # try:
    wandb.watch(conf_model)
    # except: # resuming a run
    #     pass
        
        

    # Eval and Logging
    confs = {
        opt['confidence_method']: conf_model,
    } 
    if opt['confidence_method'] == 'oec':
        confs['ed'] = FSCConfidence(fs_model, 'ed')
    elif opt['confidence_method'] == 'deep-oec':
        encoder = fs_model.encoder
        deep_mahala_obj = DeepMahala(None, None, None, encoder, device,num_feats=encoder.depth, num_classes=train_data['n_classes'], pretrained_path="", fit=False, normalize=None)
        confs['dm'] = DMConfidence(deep_mahala_obj, {'ls':range(encoder.depth),'reduction':'max','g_magnitude':0}, True, 'iso').to(device)
    # Temporal Ensemble for Evaluation
    if opt['n_ensemble'] > 1:
        nets = [deepcopy(conf_model) for _ in range(opt['n_ensemble'])]
        confs['mixture-'+opt['confidence_method']]  = Ensemble(nets, 'mixture')
        confs['poe-'+opt['confidence_method']]  = Ensemble(nets, 'poe')
        ensemble_update_interval = opt['eval_every_outer']//opt['n_ensemble']
            
    iteration_fieldnames = ['global_iteration'] 
    for c in confs:
        iteration_fieldnames += [f'{c}_train_ooe', f'{c}_val_ooe',f'{c}_test_ooe', f'{c}_ood']
    iteration_logger = CSVLogger(every=0,
                                 fieldnames=iteration_fieldnames,
                                 filename=os.path.join(opt['output_dir'], 'iteration_log.csv'))

    best_val_ooe = 0
    PATIENCE = 5 # Number of evaluations to wait
    waited = 0

    progress_bar = tqdm(range(start_idx, opt['train_iter']))
    for outer_idx in progress_bar:
        sample = load_episode(train_data, tr, opt['data.test_way'], opt['data.test_shot'], opt['data.test_query'], device)

        conf_model.train()
        if opt['full_supervision']: # sanity check
            conf_model.support(sample['xs'])
            in_score = conf_model.score(sample['xq'], detach=False).squeeze()
            out_score= conf_model.score(sample['ooc_xq'], detach=False).squeeze()
            out_scores = [out_score]
            for curr_ood, ood_tensor in ood_tensors:
                if curr_ood == 'ooe':
                    continue
                start = outer_idx % (len(ood_tensor)//2)
                stop  = min(start+sample['xq'].shape[0]*sample['xq'].shape[0], len(ood_tensor)//2)
                oxq = torch.stack([tr(x) for x  in ood_tensor[start:stop]]).to(device)
                o = conf_model.score(oxq, detach=False).squeeze()
                out_scores.append(o)
            #
            out_score = torch.cat(out_scores)
            in_score = in_score.repeat(len(ood_tensors))
            loss, acc = compute_loss_bce(in_score, out_score, mean_center=False)
        else:
            conf_model.support(sample['xs'])
            if opt['interpolate']:
                half_n_way = sample['xq'].shape[0] // 2
                interp = .5 * (sample['xq'][:half_n_way] + sample['xq'][half_n_way:2*half_n_way])
                sample['ooc_xq'][:half_n_way] = interp

            if opt['input_regularization'] == 'oe':
                # Reshape ooc_xq
                nw, nq, c, h, w = sample['ooc_xq'].shape
                sample['ooc_xq'] = sample['ooc_xq'].view(1,nw*nq, c, h, w)
                oe_bs = int(nw*nq * opt['input_regularization_percent'])

                start = (outer_idx * oe_bs) % len(reg_data)
                end = np.min([start+oe_bs,  len(reg_data)])
                oe_batch = torch.stack([tr(x) for x  in reg_data[start:end]]).to(device)
                oe_batch = oe_batch.unsqueeze(0)
                sample['ooc_xq'][:,:oe_batch.shape[1]] = oe_batch


            if opt['in_out_1_batch']:
                inps = torch.cat([sample['xq'], sample['ooc_xq']], 1)
                scores = conf_model.score(inps, detach=False).squeeze()
                in_score, out_score = scores[:sample['xq'].shape[1]], scores[sample['xq'].shape[1]:]
            else:
                in_score = conf_model.score(sample['xq'], detach=False).squeeze()
                out_score= conf_model.score(sample['ooc_xq'], detach=False).squeeze()
            
            loss, acc = compute_loss_bce(in_score, out_score, mean_center=False)

        if conf_model.pretrain_parameters() is not None and outer_idx < pretrain_iter:
            pretrain_optimizer.zero_grad()
            loss.backward()
            pretrain_optimizer.step()
        else:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        progress_bar.set_postfix(loss='{:.3e}'.format(loss), acc='{:.3e}'.format(acc))

        # Update Ensemble
        if opt['n_ensemble'] > 1 and outer_idx % ensemble_update_interval == 0:
            update_ind = (outer_idx // ensemble_update_interval) % opt['n_ensemble']
            if opt['db']:
                print(f"===> Updating Ensemble: {update_ind}")
            confs['mixture-'+opt['confidence_method']].nets[update_ind] = deepcopy(conf_model)
            confs['poe-'+opt['confidence_method']].nets[update_ind] = deepcopy(conf_model)

        # AUROC eval
        if outer_idx % opt['eval_every_outer'] == 0:
            if not opt['eval_in_train']:
                conf_model.eval()
            

            # Eval..
            stats_dict = { 'global_iteration': outer_idx }
            for conf_name, conf in confs.items():
                conf.eval()
                # OOE eval
                ooe_aurocs = {}
                for split, in_data in [('train',train_data), ('val',val_data), ('test',test_data)]:
                    auroc = np.mean(
                        eval_ood_aurocs(
                          None,
                          in_data,
                          tr, 
                          opt['data.test_way'],
                          opt['data.test_shot'],
                          opt['data.test_query'],
                          opt['data.test_episodes'],
                          device,
                          conf,
                          no_grad = False if opt['confidence_method'].startswith('dm') else True
                          )['aurocs']
                        )
                    ooe_aurocs[split]= auroc
                    print_str = '{}, iter: {} ({}), auroc: {:.3e}'.format(conf_name, outer_idx, split, ooe_aurocs[split])
                    _print_and_log(print_str, trace_file)
                stats_dict[f'{conf_name}_train_ooe'] = ooe_aurocs['train']
                stats_dict[f'{conf_name}_val_ooe'] = ooe_aurocs['val']
                stats_dict[f'{conf_name}_test_ooe'] = ooe_aurocs['test']
                
                # OOD eval
                if not opt['ooe_only']:
                    aurocs = []
                    for curr_ood, ood_tensor in ood_tensors:
                        auroc = np.mean(
                            eval_ood_aurocs(
                              ood_tensor,
                              test_data,
                              tr, 
                              opt['data.test_way'],
                              opt['data.test_shot'],
                              opt['data.test_query'],
                              opt['data.test_episodes'],
                              device,
                              conf,
                              no_grad = False if opt['confidence_method'].startswith('dm') else True
                              )['aurocs']
                            )
                        aurocs.append(auroc)

                        print_str = '{}, iter: {} ({}), auroc: {:.3e}'.format(conf_name, outer_idx, curr_ood, auroc)
                        _print_and_log(print_str, trace_file)

                    mean_ood_auroc = np.mean(aurocs)
                    print_str = '{}, iter: {} (OOD_mean), auroc: {:.3e}'.format(conf_name, outer_idx, mean_ood_auroc)
                    _print_and_log(print_str, trace_file)

                    stats_dict[f'{conf_name}_ood'] = mean_ood_auroc

                    
            iteration_logger.writerow(stats_dict)
            plot_csv(iteration_logger.filename,iteration_logger.filename)
            wandb.log(stats_dict)

            if stats_dict[f'{opt["confidence_method"]}_val_ooe'] > best_val_ooe:
                conf_model.cpu()
                torch.save(conf_model.state_dict(), os.path.join(opt['output_dir'], opt['exp_name']+'_conf_best.pt'))
                conf_model.to(device)
                # Ckpt ensemble
                if opt['n_ensemble'] >1:
                    ensemble = confs['mixture-'+opt['confidence_method']]
                    ensemble.cpu()
                    torch.save(ensemble.state_dict(), os.path.join(opt['output_dir'], opt['exp_name']+'_ensemble_best.pt'))
                    ensemble.to(device)
                waited = 0
            else:
                waited += 1
                if waited >= PATIENCE:
                    print("PATIENCE exceeded...exiting")
                    sys.exit()
            # For `resume`
            conf_model.cpu()
            torch.save({
                'conf_model_sd': conf_model.state_dict(),
                'optimizer' : optimizer,
                'pretrain_optimizer' :  pretrain_optimizer if conf_model.pretrain_parameters() is not None else None,
                'scheduler' : scheduler,
                'outer_idx': outer_idx,
                }, os.path.join(opt['output_dir'], 'last_ckpt.pt'))
            conf_model.to(device)
            conf_model.train()
    sys.exit()


if __name__ == '__main__':
    args = vars(parser.parse_args())

    if not os.path.exists(args['output_dir']):
        os.makedirs(args['output_dir'])

    with open(os.path.join(args['output_dir'], 'args.json'), 'w') as f:
        json.dump(args, f, sort_keys=True, indent=4)

    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    wandb.init(project="train_confidence")

    main(args)
