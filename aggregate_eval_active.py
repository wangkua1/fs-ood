import os
import argparse
from collections import defaultdict
import ipdb
import numpy as np
import pickle
import torch

def get_95_percent_ci(std, n):
      """Computes the 95% confidence interval from the standard deviation."""
      return std * 1.96 / n


def main(args):

    result_files = [fname for fname in os.listdir(args.exp_dir) if fname.startswith('eval_active')]
    agg_ssl_soft = defaultdict(list)
    agg_ssl_hard = defaultdict(list)
    agg_active_supervised = defaultdict(list)
    agg_active_augmented = defaultdict(list)
    for result_file in result_files:
        result = pickle.load(open( os.path.join(args.exp_dir,result_file),'rb'))
        (ssl_soft,ssl_hard,active_supervised,active_augmented) = result
        for k in ssl_soft.keys():
            agg_ssl_soft[k].append(torch.cat(ssl_soft[k],0))
            agg_ssl_hard[k].append(torch.cat(ssl_hard[k],0))
            agg_active_supervised[k].append(torch.cat(active_supervised[k],0))
            agg_active_augmented[k].append(torch.cat(active_augmented[k],0))
    # cat
    for k in ssl_soft.keys():
        agg_ssl_soft[k] = torch.cat(agg_ssl_soft[k], 0)
        agg_ssl_hard[k] = torch.cat(agg_ssl_hard[k], 0)
        agg_active_supervised[k] = torch.cat(agg_active_supervised[k], 0)
        agg_active_augmented[k] = torch.cat(agg_active_augmented[k], 0)

    N = len(agg_active_augmented[k])
    stats_ssl_soft = {}
    stats_ssl_hard = {}
    stats_active_supervised = {}
    stats_active_augmented = {}
    for k in ssl_soft.keys():
        stats_ssl_soft[k] = (agg_ssl_soft[k].mean().item() ,get_95_percent_ci(agg_ssl_soft[k].std().item(),N)  )
        stats_ssl_hard[k] = (agg_ssl_hard[k].mean().item() ,get_95_percent_ci(agg_ssl_hard[k].std().item(),N)  )
        stats_active_supervised[k] = (agg_active_supervised[k].mean().item() ,get_95_percent_ci(agg_active_supervised[k].std().item(),N)  )
        stats_active_augmented[k] = (agg_active_augmented[k].mean().item() ,get_95_percent_ci(agg_active_augmented[k].std().item(),N)  )   
    for name, dic in zip(['only_labeled','ssl_no_acq',f'ssl_{args.f_acq}',f'active_{args.f_acq}'],
                         [stats_active_supervised, stats_ssl_soft, stats_ssl_hard,stats_active_augmented]):
        with open(os.path.join(args.exp_dir,f'{name}.txt'), 'w' ) as f:
            for k in dic.keys():
                f.write(k+',')
                # Results using the support only.
                # ipdb.set_trace()
                f.write(','.join([f"{item:.3f}" for item in  dic[k]]))
                f.write('\n')

if __name__ == '__main__':
    default_exp_dir = ""
    parser = argparse.ArgumentParser(description='Plot Semi-Supervised Learning Results')
    parser.add_argument('--exp_dir', type=str, default=default_exp_dir,
                        help='Experiment directory containing result logs to plot')
    parser.add_argument('--f_acq', type=str, default='')
    args = parser.parse_args()

    assert args.f_acq!=''
    main(args)