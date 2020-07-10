import os
import ipdb
import pickle

import numpy as np
import sklearn.metrics as sk

import torch
import torchvision
import torchvision.transforms as transforms

DATA_DIR='./data'
recall_level_default = 0.9


def show_ood_detection_results_softmax(in_score, out_score):
    """show_ood_detection_results_softmax

    Args:
        in_score: shape (batch_size, ) the score of in-distribution data
        out_score: shape (batch_size, ) the score of ood data ** ideally, in_score > out_score
    Return:
        normality_base_rate: [0,1], len(in_score)/total length
        auroc: [0,1] area under ROC (higer is better)
        n_aupr: [0,1] area under P/R curve (higer is better)
        ab_aupr: [0, 1] are under P/R curve in-/out- flipped (higer is better)
    """

    normality_base_rate = round(100*in_score.shape[0] / (out_score.shape[0] + in_score.shape[0]), 2)
    # Normality Detection
    labels = np.zeros((in_score.shape[0] + out_score.shape[0]), dtype=np.int32)
    labels[:in_score.shape[0]] += 1
    examples = np.hstack((in_score, out_score))
    n_aupr = round(100*sk.average_precision_score(labels, examples), 1)
    auroc = round(100*sk.roc_auc_score(labels, examples), 1)
    # FPR90
    fpr = round(100*fpr_and_fdr_at_recall(labels, examples), 1)

    # Abnormality Detection
    in_score, out_score = -in_score, -out_score
    labels = np.zeros((in_score.shape[0] + out_score.shape[0]), dtype=np.int32)
    labels[in_score.shape[0]:] += 1
    examples = np.hstack((in_score, out_score))
    ab_aupr = round(100*sk.average_precision_score(labels, examples), 1)
    return normality_base_rate, auroc, n_aupr, ab_aupr, fpr


def _load_ood_dataset(ood_dataset_name, opt_config):
    if ood_dataset_name == 'notMNIST':
        # N_ANOM = 2000
        pickle_file = os.path.join(DATA_DIR, 'notMNIST.pickle')
        with open(pickle_file, 'rb') as f:

            try:
                save = pickle.load(f, encoding='latin1')
            except TypeError:
                save = pickle.load(f)

            ood_data = save['train_dataset'][:,None] * opt_config['ood_scale']  # (20000, 1, 28, 28)
            del save  # Is this necessary? Does it do anything?
        return ood_data
    elif ood_dataset_name == 'cifar10bw':
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((28,28)),  # Resized to 28x28 to match the size of Omniglot digits
            transforms.ToTensor(),
        ])

        cifar10_batch_size = 10
        cifar10_testset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform)
        cifar10_testloader = torch.utils.data.dataloader.DataLoader(cifar10_testset, batch_size=cifar10_batch_size, shuffle=False)
        cifar10_testiter = iter(cifar10_testloader)

        ood_data_list = []

        while True:
            try:
                cifar10_images, _ = cifar10_testiter.next()
                ood_data_list.append(cifar10_images)
            except StopIteration:
                break

        ood_data = torch.cat(ood_data_list, 0)
        return ood_data.numpy() * opt_config['ood_scale']  # For consistency, all parts of this function return numpy arrays  (10000, 1, 28, 28)
    elif ood_dataset_name == 'gaussian':
        return np.clip(.5 + np.random.normal(size=(opt_config['n_anom'], 3, 28, 28)), a_min=0, a_max=1)
    elif ood_dataset_name == 'uniform':
        return np.random.uniform(size=(opt_config['n_anom'], 1, 28, 28)) * opt_config['ood_scale']
    elif ood_dataset_name == 'rademacher':
        return (np.random.binomial(1, .5, size=(opt_config['n_anom'], 3, 32, 32)))
    elif ood_dataset_name == 'texture3':
        return torch.load(os.path.join(DATA_DIR, 'dtd.t7')).numpy() / 255.
    elif ood_dataset_name == 'places3':
        return torch.load(os.path.join(DATA_DIR, 'places.t7')).numpy() / 255.
    elif ood_dataset_name == 'svhn':
        ds = torchvision.datasets.SVHN('data', split='test', transform=None, target_transform=None, download=True)
        data = ds.data
        np.random.shuffle(data)
        return data[:10000] / 255.
    # LSUN, iSUN, and TinyImageNet are based on ODIN: https://github.com/facebookresearch/odin/blob/master/code/cal.py
    elif ood_dataset_name == 'lsun':
        transform = transforms.Compose([transforms.ToTensor()])
        ood_data = torchvision.datasets.ImageFolder(os.path.join(DATA_DIR, 'LSUN'), transform=transform)
        return np.stack([img.numpy() for (img, label) in ood_data])
    elif ood_dataset_name == 'lsun_resized':
        transform = transforms.Compose([transforms.ToTensor()])
        ood_data = torchvision.datasets.ImageFolder(os.path.join(DATA_DIR, 'LSUN_resize'), transform=transform)
        return np.stack([img.numpy() for (img, label) in ood_data])
    elif ood_dataset_name == 'isun':
        transform = transforms.Compose([transforms.ToTensor()])
        ood_data = torchvision.datasets.ImageFolder(os.path.join(DATA_DIR, 'iSUN'), transform=transform)
        return np.stack([img.numpy() for (img, label) in ood_data])
    elif ood_dataset_name == 'tinyimagenet':
        transform = transforms.Compose([transforms.ToTensor()])
        ood_data = torchvision.datasets.ImageFolder(os.path.join(DATA_DIR, 'Imagenet'), transform=transform)
        return np.stack([img.numpy() for (img, label) in ood_data])
    elif ood_dataset_name == 'tinyimagenet_resized':
        transform = transforms.Compose([transforms.ToTensor()])
        ood_data = torchvision.datasets.ImageFolder(os.path.join(DATA_DIR, 'Imagenet_resize'), transform=transform)
        return np.stack([img.numpy() for (img, label) in ood_data])
    elif ood_dataset_name == 'tinyimages':
        loaded = pickle.load(open(os.path.join(DATA_DIR, 'tinyimages_50000.pkl'),'rb'))
        return np.transpose(np.array(loaded['images']), (0,3,1,2)) / 255
    elif ood_dataset_name == 'cifar-fs-train-test':
        cifar = torchvision.datasets.CIFAR100(DATA_DIR, train=False, transform=None)
        return np.transpose(cifar.data[np.array(cifar.targets)<64], (0,3,1,2)) / 255
    elif ood_dataset_name == 'cifar-fs-test':
        cifar = torchvision.datasets.CIFAR100(DATA_DIR, train=False, transform=None)
        return np.transpose(cifar.data[np.array(cifar.targets)>=80], (0,3,1,2)) / 255
    else:
        raise ValueError('invalid OOD type')
   
        
        


def load_ood_data(ooc_config):
    # Note:
    # Most of the time, the test set of our in-distribution has about 10k
    # examples.  This was a choice made while preparing the OOD datasets.
    # So, datasets like 'texture3' which requires the users to download a
    # .t7 files has only 10k examples.

    # Load OOD dataset
    ood_dataset = _load_ood_dataset(ooc_config['name'], ooc_config)[:ooc_config['n_anom']]
    ood_dataset = np.transpose(ood_dataset,(0,2,3,1)) # (B, 3, H, W) -> (B, H, W, 3)
    assert ood_dataset.max() <= 1
    ood_dataset = (ood_dataset * 255).astype('uint8')    
    assert (ood_dataset.shape[3] in [1,3])
    return ood_dataset


## below copied from
## https://github.com/hendrycks/outlier-exposure/blob/master/utils/display_results.py

def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out


# fpr == false positive rate.
def fpr_and_fdr_at_recall(y_true, y_score, recall_level=recall_level_default, pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                 np.array_equal(classes, [-1, 1]) or
                 np.array_equal(classes, [0]) or
                 np.array_equal(classes, [-1]) or
                 np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))

if __name__ == '__main__':
    out_list = ['gaussian', 'rademacher', 'texture3', 'svhn','tinyimagenet','lsun']
    ood_tensors = [load_ood_data({
                      'name': out_name,
                      'ood_scale': 1,
                      'n_anom': 100,
                    }) for out_name in out_list]
    import ipdb; ipdb.set_trace()

