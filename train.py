"""Train a few-shot classifier.
"""
import os
import ipdb
import json
import argparse

import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import torchnet as tnt
from torchvision import transforms

from csv_logger import CSVLogger, plot_csv
import protonet
from data import get_dataset, get_transform, load_episode
from classify_classic import ResNetClassifier
# from basetrain import BaselineTrain

parser = argparse.ArgumentParser(description='Train prototypical networks')
# data args
parser.add_argument('--data.dataset', type=str, default='omniglot', metavar='DS',
                    help="data set name (default: omniglot)")
parser.add_argument('--data.split', type=str, default='vinyals', metavar='SP',
                    help="split name (default: vinyals)")
parser.add_argument('--data.cifar100_train_test', type=str, default='both', choices=['train','test', 'both'])
parser.add_argument('--data.way', type=int, default=60, metavar='WAY',
                    help="number of classes per episode (default: 60)")
parser.add_argument('--data.num_distractors', type=int, default=0, metavar='NUMDISTRACTORS',
                    help="number of distractor classes per episode (default: 0)")
parser.add_argument('--data.shot', type=int, default=5, metavar='SHOT',
                    help="number of support examples per class (default: 5)")
parser.add_argument('--data.query', type=int, default=5, metavar='QUERY',
                    help="number of query examples per class (default: 5)")
parser.add_argument('--data.unlabeled', type=int, default=0, metavar='UNLABELED',
                    help="number of unlabeled examples per class (default: 0)")
parser.add_argument('--data.test_way', type=int, default=5, metavar='TESTWAY',
                    help="number of classes per episode in test. 0 means same as data.way (default: 5)")
parser.add_argument('--data.test_num_distractors', type=int, default=0, metavar='TESTNUMDISTRACTORS',
                    help="number of distractor classes per episode in test. (default: 0)")
parser.add_argument('--data.test_shot', type=int, default=0, metavar='TESTSHOT',
                    help="number of support examples per class in test. 0 means same as data.shot (default: 0)")
parser.add_argument('--data.test_query', type=int, default=15, metavar='TESTQUERY',
                    help="number of query examples per class in test. 0 means same as data.query (default: 15)")
parser.add_argument('--data.test_unlabeled', type=int, default=0, metavar='TESTUNLABELED',
                    help="number of unlabeled examples per class in test. (default: 0)")
parser.add_argument('--data.train_episodes', type=int, default=100, metavar='NTRAIN',
                    help="number of train episodes per epoch (default: 100)")
parser.add_argument('--data.test_episodes', type=int, default=100, metavar='NTEST',
                    help="number of test episodes per epoch (default: 100)")
parser.add_argument('--data.sequential', action='store_true', default=False,
                    help="use sequential sampler instead of episodic (default: False)")
parser.add_argument('--data.cuda', action='store_true', default=True,
                    help="run in CUDA mode (default: True)")
parser.add_argument('--data.ooc_name', type=str, default='ooe', metavar='OOC',
                    help="{'ooe', 'tinyimages', 'notMNIST', 'cifar10bw', 'gaussian', 'uniform', 'MNIST'}")
parser.add_argument('--data.label_percentage', type=float, default=1.0, metavar='LABELPERCENTAGE',
                    help="The percentage of examples of each class to be treated as labeled data. (default: all of them). Used for semi-supervised experiments on tiered-ImageNet.")

# model args
parser.add_argument('--model.model_path', type=str, default='', metavar='MODELPATH',
                    help="location of pretrained model to evaluate. ")
parser.add_argument('--model.encoder', type=str, default='conv4', metavar='MODEL',
                    help="{conv4, conv4bw, conv4bbb}")
parser.add_argument('--model.f_acq', type=str, default='spp', metavar='ACQ',
                    help="{'spp', 'ed', 'oec'}")
parser.add_argument('--model.ooe_lambda', type=float, default=0, metavar='OOE',
                    help='ooe_loss weight (default: 0)')
parser.add_argument('--model.z_dim', type=int, default=64, metavar='ZDIM',
                    help="dimensionality of output images (default: 64)")
parser.add_argument('--model.method', type=str, default='baseline', metavar='METHOD',
                    help='Method to use (e.g., odin, outlier_exposure)')
parser.add_argument('--model.oe_lambda', type=float, default=0.5, metavar='OE_LAMBDA',
                    help='Outlier exposure lambda (default 0.5)')
parser.add_argument('--model.use_support_stats', type=int, default=1)
# OEC hyperparameters:
parser.add_argument('--model.lcbo_arch', type=str, default='500,500',
                    help='OEC hidden layers')
parser.add_argument('--model.lcbo_aggregation', type=str, default='max',
                    help='OEC aggregation function (max or mean)')
# MAML
parser.add_argument('--model.class', type=str, default='protonet', choices=['protonet', 'maml', 'abml'])

# train args
parser.add_argument('--train.epochs', type=int, default=500, metavar='NEPOCHS',
                    help='number of epochs to train (default: 500)')
parser.add_argument('--train.optim_method', type=str, default='Adam', metavar='OPTIM',
                    help='optimization method (default: Adam)')
parser.add_argument('--train.learning_rate', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--train.decay_every', type=int, default=10000, metavar='LRDECAY',
                    help='number of epochs after which to decay the learning rate')
parser.add_argument('--train.weight_decay', type=float, default=5e-5, metavar='WD',
                    help="weight decay (default: 5e-5)")
parser.add_argument('--train.patience', type=int, default=200, metavar='PATIENCE',
                    help='number of epochs to wait before validation improvement (default: 200)')
parser.add_argument('--train.best_metric', type=str, default='acc',
                    help='Metric used to decide when to save ckpt'
                         "{'acc', 'ooe_auroc', ''}, '' saves the last ckpt")

parser.add_argument('--data_augmentation', type=int, default=0,
                    help='augment data by flipping and cropping')
# MAML
parser.add_argument('--maml.task_update_num', type=int, default=5)
# ABML
parser.add_argument('--abml.n_infer_samples', type=int, default=2)
parser.add_argument('--abml.inner_kl_lambda', type=float, default=1)
parser.add_argument('--abml.outer_kl_lambda', type=float, default=1)
parser.add_argument('--abml.gamma_a', type=float, default=2.)
parser.add_argument('--abml.gamma_b', type=float, default=.2)
parser.add_argument('--abml.sample_q_without_sgd', action='store_true')
parser.add_argument('--abml.outer_loss_query_and_support', action='store_true')
parser.add_argument('--abml.outer_loss_use_kl_phi', action='store_true')


# log args
default_fields = 'loss,acc,class_loss,ooe_loss,ooe_auroc'
parser.add_argument('--log.fields', type=str, default=default_fields, metavar='FIELDS',
                    help="fields to monitor during training (default: {:s})".format(default_fields))
parser.add_argument('--log.exp_dir', type=str, default='results', metavar='EXP_DIR',
                    help="directory where experiments should be saved (default: results/)")
parser.add_argument('--seed', type=int, default=1234, metavar='SEED',
                    help='Set the random seed')
parser.add_argument('--ckpt_every', type=int, default=50)
parser.add_argument('--train_baseline', type=int, default=0)
parser.add_argument('--dataroot', type=str, default=os.path.join(os.environ['ROOT1'], 'data'))
args = vars(parser.parse_args())

def classification_accuracy(sample, model):
    lpy_dic = model.log_p_y(sample['xs'], sample['xq'],no_grad=True)
    log_p_y, target_inds = lpy_dic['log_p_y'], lpy_dic['target_inds']
    conf, y_hat = log_p_y.max(-1)
    corrects = torch.eq(y_hat, target_inds.squeeze()).float().view(-1)
    confs = torch.exp(conf).view(-1)
    return corrects ,confs

def main(args):

    device = 'cuda:0' if args['data.cuda'] else 'cpu'

    args['log.exp_dir'] = args['log.exp_dir']

    if not os.path.isdir(args['log.exp_dir']):
        os.makedirs(args['log.exp_dir'])

    # save opts
    with open(os.path.join(args['log.exp_dir'], 'args.json'), 'w') as f:
        json.dump(args, f)
        f.write('\n')

    # Loggin
    iteration_fieldnames = ['global_iteration', 'val_acc'] 
    iteration_logger = CSVLogger(every=0,
                                 fieldnames=iteration_fieldnames,
                                 filename=os.path.join(args['log.exp_dir'], 'iteration_log.csv'))

    # Set the random seed manually for reproducibility.
    np.random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    if args['data.cuda']:
        torch.cuda.manual_seed(args['seed'])

    if args['data.dataset'] == 'omniglot':
        raise
        train_tr = None
        test_tr = None
    elif args['data.dataset'] == 'miniimagenet':
        train_data = get_dataset('miniimagenet-train-train', args['dataroot'])
        val_data = get_dataset('miniimagenet-val', args['dataroot'])
        train_tr = get_transform('cifar_augment_normalize_84' if args['data_augmentation'] else 'cifar_normalize')
        test_tr = get_transform('cifar_normalize')

    elif args['data.dataset'] == 'cifar100':
        train_data = get_dataset('cifar-fs-train-train')
        val_data = get_dataset('cifar-fs-val')
        train_tr = get_transform('cifar_augment_normalize' if args['data_augmentation'] else 'cifar_normalize')
        test_tr = get_transform('cifar_normalize')
    else:
        raise 


    model = protonet.create_model(**args)

    if args['model.model_path'] != '':
        loaded = torch.load(args['model.model_path'])
        if not 'Protonet' in str(loaded.__class__):
            pretrained = ResNetClassifier(64, train_data['im_size']).to(device)
            pretrained.load_state_dict(loaded)
            model.encoder = pretrained.encoder
        else:
            model = loaded
        

    model = model.to(device)

    max_epoch = args['train.epochs']
    epoch = 0
    stop = False
    patience_elapsed = 0
    best_metric_value = 0.0

    def evaluate():
        
        nonlocal best_metric_value
        nonlocal patience_elapsed
        nonlocal stop
        nonlocal epoch

        corrects = []
        for _ in tqdm(range(args['data.test_episodes']), desc="Epoch {:d} Val".format(epoch + 1)):
            sample = load_episode(val_data, test_tr, args['data.test_way'], args['data.test_shot'], args['data.test_query'], device)
            corrects.append(classification_accuracy(sample, model)[0])
        val_acc = torch.mean(torch.cat(corrects))
        iteration_logger.writerow({
            'global_iteration': epoch,
            'val_acc': val_acc.item()
            })
        plot_csv(iteration_logger.filename,iteration_logger.filename)
            

        print(f"Epoch {epoch}: Val Acc: {val_acc}")
        
        if val_acc > best_metric_value:
            best_metric_value = val_acc
            print("==> best model (metric = {:0.6f}), saving model...".format(best_metric_value))
            model.cpu()
            torch.save(model, os.path.join(args['log.exp_dir'], 'best_model.pt'))
            model.to(device)
            patience_elapsed = 0

        else:
            patience_elapsed += 1
            if patience_elapsed > args['train.patience']:
                print("==> patience {:d} exceeded".format(args['train.patience']))
                stop = True


    optim_method = getattr(optim, args['train.optim_method'])
    params = model.parameters()

    optimizer = optim_method(params, lr=args['train.learning_rate'], weight_decay=args['train.weight_decay'])


    scheduler = lr_scheduler.StepLR(optimizer, args['train.decay_every'], gamma=0.5)


    while epoch < max_epoch and not stop:
        evaluate()

        model.train()
        if epoch % args['ckpt_every'] == 0:
            model.cpu()
            torch.save(model, os.path.join(args['log.exp_dir'], f'model_{epoch}.pt'))
            model.to(device)

        scheduler.step()

        for _ in tqdm(range(args['data.train_episodes']), desc="Epoch {:d} train".format(epoch + 1)):
            sample = load_episode(train_data, train_tr, args['data.way'], args['data.shot'], args['data.query'], device)
            optimizer.zero_grad()
            loss, output = model.loss(sample)
            loss.backward()
            optimizer.step()

        epoch += 1

        

if __name__ == '__main__':

    try:
        if args['data.dataset'] == 'omniglot':
            args['model.idim'] = 64
            args['data.x_dim'] = [1,28,28]
        elif args['data.dataset'] == 'cifar100':
            args['model.idim'] = 256
            args['data.x_dim'] = [3,32,32]
        elif args['data.dataset'] == 'miniimagenet':
            args['model.idim'] = 1600
            args['data.x_dim'] = [3,84,84]
        elif args['data.dataset'] == 'tieredimagenet':
            args['model.idim'] = 1600
            args['data.x_dim'] = [3,84,84]

        args['model.lcbo_arch'] = list(map(int, args['model.lcbo_arch'].split(',')))
        # if args['train_baseline']:
        #     train_baseline(args)
        # else:
        main(args)

    except KeyboardInterrupt:
        print('=' * 80)
        print('Exiting training!')
        print('=' * 80)
