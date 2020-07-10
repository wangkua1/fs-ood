import torch
from torchvision import datasets, transforms
import os
import numpy as np
from copy import deepcopy
import ipdb
from glow.datasets import  preprocess as glow_preproc
import pickle as pkl

DOWNLOAD=True
def load_episode(dataset, tr, n_way, n_shot, n_query, device):
    assert n_way > 0 and n_shot > 0 and n_query > 0
    sample = {}
    all_ys = list(set(dataset['y']))
    np.random.shuffle(all_ys)
    in_ys = all_ys[:n_way]
    out_ys = all_ys[n_way:2*n_way]
    xq, xs, ooc_xq = [], [], []
    for y in in_ys:
        x_y = dataset['x'][dataset['y'] == y]
        np.random.shuffle(x_y)
        xs.append(torch.stack([tr(x) for x in x_y[:n_shot]]))
        xq.append(torch.stack([tr(x) for x in x_y[n_shot:n_shot+n_query]]))
    for y in out_ys:
        x_y = dataset['x'][dataset['y'] == y]
        np.random.shuffle(x_y)
        ooc_xq.append(torch.stack([tr(x) for x in x_y[:n_query]]))

    xs = torch.stack(xs).to(device)
    xq = torch.stack(xq).to(device)
    ooc_xq = torch.stack(ooc_xq).to(device)
    assert xq.shape == ooc_xq.shape
    return {
        'xs': xs,
        'xq': xq,
        'ooc_xq': ooc_xq,
    }

def shuffle_data_order(data):
    inds = np.random.permutation(len(data['x']))
    data['x'] = data['x'][inds]
    data['y'] = data['y'][inds]
    return data

def _load_cached_miniimagenet(cache_path):
    try:
        with open(cache_path, "rb") as f2:
            images = pkl.load(f2, encoding='bytes')
            img_data_ = images[b'image_data']
            class_dict = images[b'class_dict']
    except:
        with open(cache_path, "rb") as f2:
            images = pkl.load(f2)
            img_data_ = images['image_data']
            class_dict = images['class_dict']
    return img_data_, class_dict

def get_dataset(dataset,  dataroot=os.path.join(os.environ['ROOT1'],'data')):
    keys = ['x', 'y', 'n_channels', 'n_classes', 'im_size']
    ret_dict = {} 
    if dataset == 'mnist-train':
        mnist = datasets.MNIST(dataroot, train=True,download=DOWNLOAD, transform=None)
        ret_dict['x'] = mnist.data
        ret_dict['y'] = mnist.targets 
        ret_dict['n_channels'] = 1
        ret_dict['n_classes'] = 10
        ret_dict['im_size'] = 28
    elif dataset == 'mnist-test':
        mnist = datasets.MNIST(dataroot, train=False, download=DOWNLOAD, transform=None)
        ret_dict['x'] = mnist.data
        ret_dict['y'] = mnist.targets 
        ret_dict['n_channels'] = 1
        ret_dict['n_classes'] = 10
        ret_dict['im_size'] = 28
    elif dataset == 'cifar10-train':
        cifar = datasets.CIFAR10(dataroot, train=True, download=DOWNLOAD, transform=None)
        ret_dict['x'] = cifar.data
        ret_dict['y'] = cifar.targets 
        ret_dict['n_channels'] = 3
        ret_dict['n_classes'] = 10
        ret_dict['im_size'] = 32
    elif dataset == 'cifar10-test':
        cifar = datasets.CIFAR10(dataroot, train=False, download=DOWNLOAD, transform=None)
        ret_dict['x'] = cifar.data
        ret_dict['y'] = cifar.targets 
        ret_dict['n_channels'] = 3
        ret_dict['n_classes'] = 10
        ret_dict['im_size'] = 32
    elif dataset == 'cifar100-train':
        cifar = datasets.CIFAR100(dataroot, train=True, download=DOWNLOAD, transform=None)
        ret_dict['x'] = cifar.data
        ret_dict['y'] = cifar.targets 
        ret_dict['n_channels'] = 3
        ret_dict['n_classes'] = 100
        ret_dict['im_size'] = 32
    elif dataset == 'cifar100-test':
        cifar = datasets.CIFAR100(dataroot, train=False, download=DOWNLOAD, transform=None)
        ret_dict['x'] = cifar.data
        ret_dict['y'] = cifar.targets 
        ret_dict['n_channels'] = 3
        ret_dict['n_classes'] = 100
        ret_dict['im_size'] = 32
    elif dataset == 'cifar-fs-train-train':
        cifar = datasets.CIFAR100(dataroot, train=True, download=DOWNLOAD, transform=None)
        ret_dict['x'] = cifar.data[np.array(cifar.targets)<64]
        ret_dict['y'] = np.asarray(cifar.targets)[np.array(cifar.targets)<64]
        ret_dict['n_channels'] = 3
        ret_dict['n_classes'] = 64
        ret_dict['im_size'] = 32
    elif dataset == 'cifar-fs-train-test':
        cifar = datasets.CIFAR100(dataroot, train=False, download=DOWNLOAD, transform=None)
        ret_dict['x'] = cifar.data[np.array(cifar.targets)<64]
        ret_dict['y'] = np.asarray(cifar.targets)[np.array(cifar.targets)<64]
        ret_dict['n_channels'] = 3
        ret_dict['n_classes'] = 64
        ret_dict['im_size'] = 32
    elif dataset == 'cifar-fs-test':
        cifar = datasets.CIFAR100(dataroot, train=False, download=DOWNLOAD, transform=None)
        ret_dict['x'] = cifar.data[np.array(cifar.targets)>=80]
        ret_dict['y'] = np.asarray(cifar.targets)[np.array(cifar.targets)>=80]
        ret_dict['n_channels'] = 3
        ret_dict['n_classes'] = 20
        ret_dict['im_size'] = 32
    elif dataset == 'cifar-fs-val':
        cifar = datasets.CIFAR100(dataroot, train=True, download=DOWNLOAD, transform=None)
        m = np.array(np.array(cifar.targets)<80).astype('int32') * np.array(np.array(cifar.targets)>=64).astype('int32')
        ret_dict['x'] = cifar.data[m.astype('bool')]
        ret_dict['y'] = np.asarray(cifar.targets)[m.astype('bool')]
        ret_dict['n_channels'] = 3
        ret_dict['n_classes'] = 16
        ret_dict['im_size'] = 32
    elif dataset == 'miniimagenet-train-train':
        split = 'train'
        img_data_, class_dict = _load_cached_miniimagenet(
                    os.path.join(dataroot, 'mini-imagenet', "mini-imagenet-cache-{:s}.pkl".format(split)))

        y = np.zeros((len(img_data_)), dtype='int32')
        for n, inds in enumerate(class_dict.values()):
            y[np.array(inds)] = n
        N = int(len(img_data_)*.9)
        # shuffle, so split contains every class
        np.random.seed(0)
        inds = np.random.permutation(len(img_data_))[:N]
        ret_dict['x'] = img_data_[inds]
        ret_dict['y'] = y[inds]
        ret_dict['n_channels'] = 3
        ret_dict['n_classes'] = 64
        ret_dict['im_size'] = 84

    elif dataset == 'miniimagenet-train-test':
        split = 'train'
        img_data_, class_dict = _load_cached_miniimagenet(
                    os.path.join(dataroot, 'mini-imagenet', "mini-imagenet-cache-{:s}.pkl".format(split)))

        y = np.zeros((len(img_data_)), dtype='int32')
        for n, inds in enumerate(class_dict.values()):
            y[np.array(inds)] = n
        N = int(len(img_data_)*.9)
        # shuffle, so split contains every class
        np.random.seed(0)
        inds = np.random.permutation(len(img_data_))[N:]
        ret_dict['x'] = img_data_[inds]
        ret_dict['y'] = y[inds]
        ret_dict['n_channels'] = 3
        ret_dict['n_classes'] = 64
        ret_dict['im_size'] = 84
        print(np.histogram(ret_dict['y'], bins=ret_dict['n_classes'], range=(0,ret_dict['n_classes'])))
    elif dataset == 'miniimagenet-val':
        split = 'val'
        img_data_, class_dict = _load_cached_miniimagenet(
                    os.path.join(dataroot, 'mini-imagenet', "mini-imagenet-cache-{:s}.pkl".format(split)))
        
        y = np.zeros((len(img_data_)), dtype='int32')
        for n, inds in enumerate(class_dict.values()):
            y[np.array(inds)] = n
        ret_dict['x'] = img_data_
        ret_dict['y'] = y
        ret_dict['n_channels'] = 3
        ret_dict['n_classes'] = 16
        ret_dict['im_size'] = 84
    elif dataset == 'miniimagenet-test':
        split = 'test'
        img_data_, class_dict = _load_cached_miniimagenet(
                    os.path.join(dataroot, 'mini-imagenet', "mini-imagenet-cache-{:s}.pkl".format(split)))
        
        y = np.zeros((len(img_data_)), dtype='int32')
        for n, inds in enumerate(class_dict.values()):
            y[np.array(inds)] = n
        ret_dict['x'] = img_data_
        ret_dict['y'] = y
        ret_dict['n_channels'] = 3
        ret_dict['n_classes'] = 20
        ret_dict['im_size'] = 84

    else:
        raise ValueError("unknown dataset")
    assert set(keys) == set(ret_dict.keys())
    assert isinstance(ret_dict['x'], np.ndarray) and len(ret_dict['x'].shape) == 4
    assert isinstance(ret_dict['y'], np.ndarray) and len(ret_dict['y'].shape) == 1
    return ret_dict

def assert_ndarray(x):
    assert isinstance(x, np.ndarray)
    return x

def one_to_three_channels(x):
    if x.shape[0] == 1:
        x = x.repeat(3,1,1)
    return x

# Magic
mnist_normalize = transforms.Normalize((0.1307,), (0.3081,))
cifar_normalize = transforms.Normalize(
        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
def get_transform(transform_name):
    
    #
    if transform_name == 'mnist_normalize':
        tr = transforms.Compose([assert_ndarray])
        tr.transforms.append(transforms.ToPILImage())
        tr.transforms.append(transforms.ToTensor())
        tr.transforms.append(mnist_normalize)              
    elif transform_name == 'mnist_resize_normalize':
        tr = transforms.Compose([assert_ndarray])
        tr.transforms.append(transforms.ToPILImage())
        tr.transforms.append(transforms.Resize((28,28)))
        tr.transforms.append(transforms.ToTensor())
        tr.transforms.append(lambda x: x.mean(0, keepdim=True))
        tr.transforms.append(mnist_normalize)
    elif transform_name == 'cifar_normalize':
        tr = transforms.Compose([assert_ndarray, transforms.ToTensor()])
        tr.transforms.append(cifar_normalize)
    elif transform_name == 'cifar_augment_normalize':
        tr = transforms.Compose([assert_ndarray, transforms.ToTensor()])
        tr.transforms.append(transforms.ToPILImage())
        tr.transforms.append(transforms.RandomCrop(32, padding=4))
        tr.transforms.append(transforms.RandomHorizontalFlip())
        tr.transforms.append(transforms.ToTensor())
        tr.transforms.append(cifar_normalize)
    elif transform_name == 'cifar_resize_normalize':
        tr = transforms.Compose([assert_ndarray])
        tr.transforms.append(transforms.ToPILImage()) # if input to this is tensor, it runs, but behaves differently from input being np
        tr.transforms.append(transforms.Resize((32,32)))
        tr.transforms.append(transforms.ToTensor())
        tr.transforms.append(one_to_three_channels)
        tr.transforms.append(cifar_normalize)
    elif transform_name == 'cifar_augment_normalize_84':
        tr = transforms.Compose([assert_ndarray, transforms.ToTensor()])
        tr.transforms.append(transforms.ToPILImage())
        tr.transforms.append(transforms.RandomCrop(84, padding=4))
        tr.transforms.append(transforms.RandomHorizontalFlip())
        tr.transforms.append(transforms.ToTensor())
        tr.transforms.append(cifar_normalize)
    elif transform_name == 'cifar_resize_normalize_84':
        tr = transforms.Compose([assert_ndarray])
        tr.transforms.append(transforms.ToPILImage()) # if input to this is tensor, it runs, but behaves differently from input being np
        tr.transforms.append(transforms.Resize((84,84)))
        tr.transforms.append(transforms.ToTensor())
        tr.transforms.append(one_to_three_channels)
        tr.transforms.append(cifar_normalize)
    elif transform_name == 'cifar_resize_glow_preproc':
        tr = transforms.Compose([assert_ndarray])
        tr.transforms.append(transforms.ToPILImage()) 
        tr.transforms.append(transforms.Resize((32,32)))
        tr.transforms.append(transforms.ToTensor())
        tr.transforms.append(one_to_three_channels)
        tr.transforms.append(glow_preproc)
    else:
        raise ValueError()

    return tr
