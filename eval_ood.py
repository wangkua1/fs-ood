from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from classify_classic  import ResNetClassifier
import numpy as np
import ipdb
import itertools
from tqdm import tqdm
from anomaly import load_ood_data, show_ood_detection_results_softmax
import os
from collections import defaultdict, OrderedDict

from functools import partial
import pandas
import json
from copy import deepcopy
from classify_classic import ResNetClassifier
from create_ensemble import Ensemble
from glow.model import Glow
import cv2
import torchvision
import torchvision.utils as vutils
from data import get_dataset, get_transform, load_episode, cifar_normalize
from confidence.base import BaseConfidence, FSCConfidence
from confidence.dm import DeepMahala, DMConfidence
from confidence.dkde import DKDEConfidence
from confidence.oec import OECConfidence, OEC
from confidence.deep_oec import DeepOECConfidence
from protonet import Protonet
from utils import mkdir
from collections import defaultdict
from baselinetrain import BaselineFinetune


def score_batch(conf, sample):
  im_size = list(sample['xs'].shape[2:])
  in_x = sample['xq'].reshape(-1,*im_size)
  out_x = sample['ooc_xq'].reshape(-1, *im_size)
  # Init conf with support
  conf.support(sample['xs'])
  # Compute conf scores
  all_x = torch.cat([in_x, out_x], 0)
  scores = conf.score(all_x).detach().cpu()
  in_score, out_score = scores[:len(in_x)], scores[len(in_x):]
  return in_score, out_score

def eval_ood_aurocs(
  ood_tensor, 
  episodic_in_data,
  tr, 
  n_way,
  n_shot,
  n_query,
  n_episodes,
  device,
  conf,
  db=False,
  out_name='',
  no_grad=True
  ):
  if ood_tensor is not None:
    N = n_way*n_query*n_episodes
    # repeat if necessary
    if len(ood_tensor) < N:
      ood_tensor = np.vstack([ood_tensor for _ in range(N//len(ood_tensor)+1)])[:N]
  metrics = defaultdict(list)
  for n in tqdm(range(n_episodes),desc='eval_ood_aurocs',dynamic_ncols=True):
    sample = load_episode(episodic_in_data, tr, n_way, n_shot, n_query, device)
    if ood_tensor is not None:
      bs = n_way*n_query
      sample['ooc_xq'] = torch.stack([tr(x) for x  in ood_tensor[n*bs:(n+1)*bs]]).to(device)
    with torch.set_grad_enabled( not no_grad ):
      in_score, out_score = score_batch(conf, sample)
      in_score = in_score.numpy()
      out_score = out_score.numpy()
      # if db:
      #   in_score1= conf.score(in_x).detach().cpu().numpy()
      #   out_score1=conf.score(out_x).detach().cpu().numpy()
      #   print("db --- d-score", np.sum((in_score - in_score1)**2 ), np.sum((out_score - out_score1)**2 ))
      _, auroc, _, _, fpr = show_ood_detection_results_softmax(in_score,out_score)
      metrics['aurocs'].append(auroc)
      metrics['fprs'].append(fpr)
  if db: 
    print("Avg `in_score`: ", np.mean(in_score))
    print("Avg `out_score`: ", np.mean(out_score))
    # vutils.save_image(out_x[:100], f'episodic-{out_name}.jpeg' , normalize=True, nrow=10) 
  return metrics


def mpp(model, x):
  return model(x).max(1)[0].exp()




def main():
    
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--dm_path', type=str, default='')
    parser.add_argument('--oec_path', type=str, default='')
    parser.add_argument('--episodic_ood_eval', type=int, default=0)
    parser.add_argument('--episodic_in_distr', type = str, default='meta-test', choices=['meta-test','meta-train'])
    # DM 
    parser.add_argument('--dm_g_magnitude', type=float, default=0)
    parser.add_argument('--dm_ls', type=str, default='-')
    parser.add_argument('--db', type = int, default=0)
    parser.add_argument('--tag', type = str, default='')
    parser.add_argument('--n_episodes', type = int, default=100)
    parser.add_argument('--n_ways', type = int, default=5)
    parser.add_argument('--n_shots', type = int, default=5)
    # Required
    parser.add_argument('--dataroot', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--dataset', required=True, choices=['mnist','cifar10', 'cifar100', 'cifar-fs', 'cifar-64', 'miniimagenet'])
    parser.add_argument('--ood_methods', type=str, required=True, help='comma separated list of method names e.g.,  `mpp,DM-all')
    ## Pretrained model paths 
    parser.add_argument('--fsmodel_path', required=True)
    parser.add_argument('--fsmodel_name', required=True, type=str, choices=['protonet', 'maml','baseline','baseline-pn'])
    parser.add_argument('--classifier_path', required=True)
    parser.add_argument('--glow_dir', required=True)
    parser.add_argument('--ooe_only', type=int, default=0)
    
    args = parser.parse_args()
    use_cuda = True

    mkdir(args.output_dir)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")


    
    if args.dataset  == 'mnist':
      test_data = get_dataset('mnist-test', args.dataroot)
      out_list = ['gaussian', 'rademacher', 'texture3', 'svhn', 'notMNIST']
      tr = get_transform('mnist_resize_normalize')
      
    if args.dataset.startswith('cifar'):
      out_list = ['gaussian', 'rademacher', 'texture3', 'svhn','tinyimagenet','lsun']
      # out_list = ['svhn']
      normalize = cifar_normalize
      if args.dataset == 'cifar10':
        train_data = get_dataset('cifar10-train', args.dataroot)
        test_data = get_dataset('cifar10-test', args.dataroot)
        
      if args.dataset == 'cifar100':
        train_data = get_dataset('cifar100-train', args.dataroot)
        test_data = get_dataset('cifar100-test', args.dataroot)
        
      if args.dataset == 'cifar-fs':
        train_data = get_dataset('cifar-fs-train-train', args.dataroot)
        test_data = get_dataset('cifar-fs-test', args.dataroot)
        
      if args.dataset == 'cifar-64':
        assert args.db 
        train_data = get_dataset('cifar-fs-train-train', args.dataroot)
        test_data = get_dataset('cifar-fs-train-test', args.dataroot)

      tr = get_transform('cifar_resize_glow_preproc') if args.ood_methods.split(',')[0].startswith('glow') else get_transform('cifar_resize_normalize') 
        
      
    if args.dataset  == 'miniimagenet':
      train_data = get_dataset('miniimagenet-train-train', args.dataroot)
      test_data = get_dataset('miniimagenet-test', args.dataroot)
      out_list = ['gaussian', 'rademacher', 'texture3', 'svhn','tinyimagenet','lsun']
      tr =  get_transform('cifar_resize_glow_preproc') if args.ood_methods.split(',')[0].startswith('glow') else get_transform('cifar_resize_normalize_84') 
      normalize = cifar_normalize

    # Models
    classifier = None
    glow = None
    fs_model = None
    ## FS Model
    if args.fsmodel_name in ['protonet', 'maml']:
      assert args.fsmodel_path != '-'
      fs_model = torch.load(args.fsmodel_path)
      encoder = fs_model.encoder
    ## Classifier
    elif args.fsmodel_name in ['baseline','baseline-pn'] :
      assert args.classifier_path != '-' 
      classifier = ResNetClassifier(train_data['n_classes'], train_data['im_size']).to(device)
      classifier.load_state_dict(torch.load(args.classifier_path))
      encoder = classifier.encoder
      if args.fsmodel_name == 'baseline':
        fs_model = BaselineFinetune(encoder, args.n_ways,args.n_shots,loss_type='dist')
      else:
        fs_model = Protonet(encoder)
    
    fs_model.to(device)
    fs_model.eval()
    args.num_feats = encoder.depth
    encoder.to(device)
    encoder.eval()

    if args.classifier_path != '-' and classifier is None: # for non-FS methods
      classifier = ResNetClassifier(train_data['n_classes'], train_data['im_size']).to(device)
      classifier.load_state_dict(torch.load(args.classifier_path))


    if args.glow_dir != '-':
      # Load Glow
      glow_name = list(filter( lambda s: 'glow_model' in s, os.listdir(args.glow_dir)))[0]
      with open(os.path.join(args.glow_dir ,'hparams.json')) as json_file:  
          hparams = json.load(json_file)
      # Notice Glow is 32,32,3 even for miniImageNet
      glow = Glow((32,32,3), hparams['hidden_channels'], hparams['K'], hparams['L'], hparams['actnorm_scale'],
           hparams['flow_permutation'], hparams['flow_coupling'], hparams['LU_decomposed'], train_data['n_classes'], hparams['learn_top'], hparams['y_condition'])
      glow.load_state_dict(torch.load(os.path.join(args.glow_dir, glow_name)))
      glow.set_actnorm_init()
      glow = glow.to(device)
      glow = glow.eval()

      
    # Verify Acc (just making sure models are loaded properly)
    if classifier is not None and not args.ood_methods.split(',')[0].startswith('glow'):
        preds = classifier(torch.stack([tr(x) for x  in train_data['x'][:args.test_batch_size]]).to(device)).max(-1)[1]
        print("Train Acc: ", (preds.detach().cpu().numpy()==np.array(train_data['y'])[:args.test_batch_size]).mean())
        preds = classifier(torch.stack([tr(x) for x  in test_data['x'][:args.test_batch_size]]).to(device)).max(-1)[1]
        print("Test Acc: ", (preds.detach().cpu().numpy()==np.array(test_data['y'])[:args.test_batch_size]).mean())


    # Confidence functions for OOD
    confidence_funcs = OrderedDict() # (name, (func, use_support, kwargs))
    for ood_method in args.ood_methods.split(','):
      no_grad = True

      if ood_method.startswith('DM'):
        deep_mahala_obj = DeepMahala(train_data['x'], train_data['y'], tr, encoder, device,num_feats=args.num_feats, num_classes=train_data['n_classes'], pretrained_path=args.dm_path, fit=True, normalize=normalize)

      if ood_method.startswith('deep-ed'): 
        no_grad=False
        deep_mahala_obj = DeepMahala(train_data['x'], train_data['y'], tr, encoder, device,num_feats=args.num_feats, num_classes=train_data['n_classes'], pretrained_path=args.dm_path, fit=False, normalize=normalize)

      if ood_method == 'MPP':
        confidence_funcs['MPP'] = BaseConfidence(lambda x:mpp(classifier, x))
      elif ood_method == 'Ensemble-MPP':
        nets = []
        class PModel(nn.Module):
          def __init__(self, logp_model):
            super(PModel, self).__init__()
            self.logp_model = logp_model
          def forward(self, x):
            return self.logp_model(x).exp()
            
        for i in range(5):
          _dir = os.path.dirname(args.classifier_path)
          _fname = os.path.basename(args.classifier_path)
          path = os.path.join(_dir[:-1]+f"{i}", _fname)
          model = ResNetClassifier(train_data['n_classes'], train_data['im_size'])
          model.load_state_dict(torch.load(path))
          model = PModel(model)
          model.eval() # 
          nets.append(model.to(device))
        ensemble = Ensemble(nets)
        confidence_funcs['Ensemble-MPP'] = BaseConfidence(lambda x:ensemble(x).max(-1)[0])
      elif ood_method == 'DM-last':
        confidence_funcs['DM-last'] = DMConfidence(deep_mahala_obj, {'ls':[args.num_feats - 1],'reduction':'max'}, False).to(device)
      elif ood_method == 'DM-all':
        confidence_funcs['DM-all'] = DMConfidence(deep_mahala_obj, {'ls':[i for i in range(args.num_feats)],'reduction':'max'}, False).to(device)
      elif ood_method == 'glow-ll':
        confidence_funcs['glow-ll'] = BaseConfidence(lambda x:-glow(x)[1])
      elif ood_method == 'glow-lr':
        from test_glow_ood import ll_to_png_code_ratio
        confidence_funcs['glow-lr'] = BaseConfidence(lambda x:ll_to_png_code_ratio(x, glow))
      elif ood_method == 'native-spp' and args.episodic_ood_eval:
        if args.fsmodel_name in ['maml','baseline']:
          no_grad=False
        confidence_funcs['native-spp'] = FSCConfidence(fs_model, 'spp')
      elif ood_method == 'native-ed' and args.episodic_ood_eval:
        confidence_funcs['native-ed'] = FSCConfidence(fs_model, 'ed')
      elif ood_method.startswith('deep-ed') and args.episodic_ood_eval:
        if args.dm_ls == '-':
          ls = range(args.num_feats)
        else:
          ls = [int(l) for l in args.dm_ls.split(',')]
        kwargs = {
          'ls':ls,
          'reduction':'max',
          'g_magnitude': args.dm_g_magnitude
        }
        dm_conf = DMConfidence(deep_mahala_obj, kwargs, True, ood_method.split('-')[-1])
        dm_conf.to(device)
        confidence_funcs[ood_method] = dm_conf
      elif ood_method == 'dkde' and args.episodic_ood_eval:
        confidence_funcs['dkde'] = DKDEConfidence(encoder)
      elif ood_method == 'oec' and args.episodic_ood_eval:
        oec_opt = json.load(
                open(os.path.join(os.path.dirname(args.oec_path), 'args.json'), 'r')
            )

        init_sample = load_episode(train_data, tr, oec_opt['data.test_way'], oec_opt['data.test_shot'], oec_opt['data.test_query'], device)
        if oec_opt['confidence_method'] == 'oec':
          oec_conf = OECConfidence(None, fs_model, init_sample, oec_opt)
        else:
          oec_conf = DeepOECConfidence(None, fs_model, init_sample, oec_opt)
        oec_conf.load_state_dict(
              torch.load(args.oec_path)
            )
        oec_conf.eval()
        oec_conf.to(device)
        confidence_funcs['oec'] =  oec_conf
      elif ood_method == 'oec-ensemble' and args.episodic_ood_eval: # not much more effective than 'oec'
        oec_opt = json.load(
                open(os.path.join(os.path.dirname(args.oec_path), 'args.json'), 'r')
            )
        oec_confs = []
        for e in range(5):
          init_sample = load_episode(train_data, tr, oec_opt['data.test_way'], oec_opt['data.test_shot'], oec_opt['data.test_query'], device)
          if oec_opt['confidence_method'] == 'oec':
            oec_conf = OECConfidence(None, fs_model, init_sample, oec_opt)
          else:
            oec_conf = DeepOECConfidence(None, fs_model, init_sample, oec_opt)
          # Find ckpt 
          cdir = os.path.dirname(args.oec_path)[:-1]+f"{e}"
          fname = list(filter(lambda s:s.endswith('conf_best.pt'), os.listdir(cdir)))[0]
          oec_conf.load_state_dict(
                torch.load(os.path.join(
                  cdir, fname))
              )
          oec_conf.eval()
          oec_conf.to(device)    
          oec_confs.append(oec_conf)
        confidence_funcs['oec'] =  Ensemble(oec_confs)
      else:
        raise # ood_method not implemented, or typo in name

 
    
    auroc_data = defaultdict(list)
    auroc_95ci_data = defaultdict(list)
    fpr_data = defaultdict(list)
    fpr_95ci_data = defaultdict(list)

    # Classic OOD evaluation
    if not args.episodic_ood_eval:
      for out_name in out_list:
        ooc_config = {
            'name': out_name,
            'ood_scale': 1,
            'n_anom': 5000,
            'cuda': False
        }
        ood_tensor = load_ood_data(ooc_config)
        assert len(ood_tensor) <= len(test_data['x'])
        in_scores = defaultdict(list)
        out_scores = defaultdict(list)

        with torch.no_grad():
          for i in tqdm(range(0, len(ood_tensor), args.test_batch_size)):
            stop = min(args.test_batch_size, len(ood_tensor[i:]))
            in_x = torch.stack([tr(x) for x  in test_data['x'][i:i+stop]]).to(device)
            out_x = torch.stack([tr(x) for x  in ood_tensor[i:i+stop]]).to(device)
            for c, f in confidence_funcs.items():
              in_scores[c].append(f.score(in_x))
              out_scores[c].append(f.score(out_x))
        # save ood images for debugging
        vutils.save_image(out_x[:100], f'non-episodic-{out_name}.png' , normalize=True, nrow=10) 
                
        for c in confidence_funcs:
          auroc = show_ood_detection_results_softmax(torch.cat(in_scores[c]).cpu().numpy(),torch.cat(out_scores[c]).cpu().numpy())[1]
          print(out_name, c, ': ', auroc)
          # 
          auroc_data[c].append(auroc)
        auroc_data['dset'].append(out_name)
      pandas.DataFrame(auroc_data).to_csv(os.path.join(args.output_dir,f'md_auroc_{args.ood_methods}.csv'))
    else:
      cifar_meta_train_data = get_dataset('cifar-fs-train-test', args.dataroot)
      cifar_meta_test_data = get_dataset('cifar-fs-test', args.dataroot)
      
      # OOD Eval
      if args.episodic_in_distr == 'meta-train':
        episodic_in_data = train_data
      else:
        episodic_in_data = test_data

      episodic_ood = ['ooe','cifar-fs-test', 'cifar-fs-train-test']  

      ood_tensors = [None] + [load_ood_data({
                      'name': out_name,
                      'ood_scale': 1,
                      'n_anom': 10000,
                    }) for out_name in episodic_ood[1:] + out_list]
      if args.ooe_only:
        all_oods = [('ooe', None)]
      else:
        all_oods = zip(episodic_ood + out_list, ood_tensors)
      for out_name, ood_tensor in all_oods:
        n_query = 15
        metrics_dic = defaultdict(list)
        for c, f in confidence_funcs.items():
          metrics_dic[c] = eval_ood_aurocs(
                      ood_tensor,
                      episodic_in_data,
                      tr, 
                      args.n_ways,
                      args.n_shots,
                      n_query,
                      args.n_episodes,
                      device,
                      f,
                      db=args.db,
                      out_name=out_name,
                      no_grad=no_grad
                      )
        
        for c in confidence_funcs:
          auroc = np.mean(metrics_dic[c]['aurocs'])
          auroc_95ci = np.std(metrics_dic[c]['aurocs']) * 1.96 / args.n_episodes
          auroc_data[c].append(auroc)
          auroc_95ci_data[c].append(auroc_95ci)
          print(out_name, c, 'auroc: ', auroc, ',', auroc_95ci)
          fpr = np.mean(metrics_dic[c]['fprs'])
          fpr_95ci = np.std(metrics_dic[c]['fprs']) * 1.96 / args.n_episodes
          fpr_data[c].append(fpr)
          fpr_95ci_data[c].append(fpr_95ci)
          print(out_name, c, 'fpr: ', fpr, ',', fpr_95ci)
          
        auroc_data['dset'].append(out_name)
        fpr_data['dset'].append(out_name)
        auroc_95ci_data['dset'].append(out_name)
        fpr_95ci_data['dset'].append(out_name)
      pandas.DataFrame(auroc_data).to_csv(os.path.join(args.output_dir,f'{args.tag}_episodic_{args.episodic_in_distr}_{args.dm_path.split(".")[0]}_{args.ood_methods}_auroc.csv'))
      pandas.DataFrame(fpr_data).to_csv(os.path.join(args.output_dir,f'{args.tag}_episodic_{args.episodic_in_distr}_{args.dm_path.split(".")[0]}_{args.ood_methods}_fpr.csv'))
      pandas.DataFrame(auroc_95ci_data).to_csv(os.path.join(args.output_dir,f'{args.tag}_episodic_{args.episodic_in_distr}_{args.dm_path.split(".")[0]}_{args.ood_methods}_auroc_95ci.csv'))
      pandas.DataFrame(fpr_95ci_data).to_csv(os.path.join(args.output_dir,f'{args.tag}_episodic_{args.episodic_in_distr}_{args.dm_path.split(".")[0]}_{args.ood_methods}_fpr_95ci.csv'))

if __name__ == '__main__':
    main()