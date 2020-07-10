from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import os
from utils import mkdir
from torchvision.utils import make_grid
import backbone
import ipdb
from csv_logger import CSVLogger, plot_csv
# from models.resnet import ResNet18, ResNet34
# from models.wide_resnet import WideResNet
import numpy as np
import sys
from data import get_dataset,  get_transform, shuffle_data_order
from tqdm import tqdm
import wandb
import torchvision.utils as vutils


class ResNetClassifier(nn.Module):
    def __init__(self,num_classes, im_size):
        super(ResNetClassifier, self).__init__()
        if im_size == 32:
            self.encoder = backbone.ResNet10_32(3) 
        elif im_size == 84:
            self.encoder = backbone.ResNet10_84(3) 
        else:
            raise ValueError
        self.classifier = backbone.Linear_fw(512, num_classes)
    def forward(self, x):
        x = self.classifier(self.encoder(x))
        return F.log_softmax(x, dim=1)
        
def train(args, model, device, data, tr, optimizer, epoch):
    data = shuffle_data_order(data)
    # Viz data, (for debugging data ordering)
    vutils.save_image(torch.stack([tr(x) for x in data['x'][:100]]), os.path.join(args.output_dir , f"data_e_{epoch}.jpeg"), normalize=True, nrow=10)

    model.train()
    for batch_idx, n in enumerate(range(0, len(data['x']), args.batch_size)):
        stop = min(args.batch_size, len(data['x'][n:]))
        x = torch.stack([tr(x) for x in data['x'][n:n+stop]]).to(device)
        target = torch.from_numpy(np.array(data['y'])[n:n+stop]).long().to(device)
        
        optimizer.zero_grad()
        output = model(x)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(x), len(data['x']),
                100. * batch_idx / (len(data['x']) // args.batch_size), loss.item()))


def test(args, model, device, data, tr, n_batches=-1):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        n_tested = 0
    for n in tqdm(range(0, len(data['x']), args.test_batch_size)):
            stop = min(args.test_batch_size, len(data['x'][n:]))
            x = torch.stack([tr(x) for x in data['x'][n:n+stop]]).to(device)
            target = torch.from_numpy(np.array(data['y'])[n:n+stop]).long().to(device)
            output = model(x)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            n_tested += len(x)
    test_loss /= n_tested

    print('\nAverage loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, n_tested,
        100. * correct / n_tested))
    return correct / n_tested

def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def main(args):
    #
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    set_random_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    if args.dataset == 'mnist':
        train_data = get_dataset('mnist-train',  args.dataroot)
        test_data = get_dataset('mnist-test',  args.dataroot)
        train_tr = test_tr = get_transform('mnist_normalize')

    if args.dataset == 'cifar10':
        train_tr_name = 'cifar_augment_normalize' if args.data_augmentation else 'cifar_normalize'
        train_data = get_dataset('cifar10-train',  args.dataroot)
        test_data = get_dataset('cifar10-test',  args.dataroot)
        train_tr = get_transform(train_tr_name)
        test_tr = get_transform('cifar_normalize')
        
    if args.dataset == 'cifar-fs-train':
        train_tr_name = 'cifar_augment_normalize' if args.data_augmentation else 'cifar_normalize'
        train_data = get_dataset('cifar-fs-train-train',  args.dataroot)
        test_data = get_dataset('cifar-fs-train-test',  args.dataroot)
        train_tr = get_transform(train_tr_name)
        test_tr = get_transform('cifar_normalize')

    if args.dataset == 'miniimagenet':
        train_data = get_dataset('miniimagenet-train-train', args.dataroot)
        test_data = get_dataset('miniimagenet-train-test', args.dataroot)
        train_tr = get_transform('cifar_augment_normalize_84' if args.data_augmentation else 'cifar_normalize')
        test_tr = get_transform('cifar_normalize')
    

    model = ResNetClassifier(train_data['n_classes'], train_data['im_size']).to(device)
    if args.ckpt_path != '':
        loaded = torch.load(args.ckpt_path)
        model.load_state_dict(loaded)
        ipdb.set_trace()
    if args.eval:
        acc = test(args, model, device, test_loader, args.n_eval_batches)
        print("Eval Acc: ", acc)
        sys.exit()

    # Trace logging
    mkdir(args.output_dir)
    eval_fieldnames = ['global_iteration','val_acc','train_acc']
    eval_logger = CSVLogger(every=1,
                                 fieldnames=eval_fieldnames,
                                 resume=args.resume,
                                 filename=os.path.join(args.output_dir, 'eval_log.csv'))
    wandb.run.name = os.path.basename(args.output_dir)
    wandb.run.save()
    wandb.watch(model)

    if args.optim == 'adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    elif args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=5e-4)
    if args.dataset == 'mnist':
        scheduler = StepLR(optimizer, step_size=1, gamma=.7)
    else:
        scheduler = MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)

    start_epoch = 1
    if args.resume:
        last_ckpt_path = os.path.join(args.output_dir, 'last_ckpt.pt')
        if os.path.exists(last_ckpt_path):
            loaded = torch.load(last_ckpt_path)
            model.load_state_dict(loaded['model_sd'])
            optimizer.load_state_dict(loaded['optimizer_sd'])
            scheduler.load_state_dict(loaded['scheduler_sd'])
            start_epoch = loaded['epoch']

    # It's important to set seed again before training b/c dataloading code
    # might have reset the seed.
    set_random_seed(args.seed)
    best_val = 0
    if args.db: 
        scheduler = MultiStepLR(optimizer, milestones=[1, 2, 3, 4], gamma=0.1)
        args.epochs = 5
    for epoch in range(start_epoch, args.epochs + 1):
        if epoch % args.ckpt_every == 0:
            torch.save(model.state_dict(), os.path.join(args.output_dir , f"ckpt_{epoch}.pt"))

        stats_dict = {'global_iteration':epoch}
        val = stats_dict['val_acc'] = test(args, model, device, test_data, test_tr, args.n_eval_batches)
        stats_dict['train_acc'] = test(args, model, device, train_data, test_tr, args.n_eval_batches)
        grid = make_grid(torch.stack([train_tr(x) for x in train_data['x'][:30]]), nrow=6).permute(1,2,0).numpy()
        img_dict = {"examples": [wandb.Image(grid, caption="Data batch")]}
        wandb.log(stats_dict)
        wandb.log(img_dict)
        eval_logger.writerow(stats_dict)
        plot_csv(eval_logger.filename, os.path.join(args.output_dir, 'iteration_plots.png'))

        train(args, model, device, train_data, train_tr, optimizer, epoch)
        
        scheduler.step(epoch)

        if val > best_val: 
            best_val = val
            torch.save(model.state_dict(), os.path.join(args.output_dir , f"ckpt_best.pt"))

        # For `resume`
        model.cpu()
        torch.save({
            'model_sd': model.state_dict(),
            'optimizer_sd': optimizer.state_dict(), 
            'scheduler_sd': scheduler.state_dict(), 
            'epoch': epoch + 1
            }, os.path.join(args.output_dir, "last_ckpt.pt"))
        model.to(device)

        

if __name__ == '__main__':
    wandb.init(project="example")

    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--dataroot', required=True)
    parser.add_argument('--pdim', type=int, default=128)
    parser.add_argument('--output_dir', required=True, help='')
    parser.add_argument('--dataset', required=True, choices=['mnist','cifar10','cifar-fs-train', 'miniimagenet'])
    parser.add_argument('--ckpt_every', type=int, default=1)
    parser.add_argument('--optim', type=str, default='adadelta')
    parser.add_argument('--n_eval_batches', type=int, default=-1)
    parser.add_argument('--data_augmentation', type=int, default=0)
    parser.add_argument('--ckpt_path', default='', type=str)
    parser.add_argument('--eval', default=0,type=int)
    parser.add_argument('--resume', default=0, type=int)
    parser.add_argument('--db', default=0, type=int)
    args = parser.parse_args()
    main(args)