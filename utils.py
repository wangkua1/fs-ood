import numpy as np

import torch
import os 
import matplotlib
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

def mkdir(p):
    if not os.path.exists(p):
        os.makedirs(p)

def make_error_boxes(ax, xdata, ydata, facecolor='r', edgecolor='k', alpha=0.5, width = .1):

    # Loop over data points; create box from errors at each point
    for i,( x, y) in enumerate(zip(xdata, ydata)):
        rect = Rectangle((x , 0), width, y)

        if type(alpha) == np.ndarray:
            a = alpha[i]
        else:
            a = alpha
        # Create patch collection with specified colour/alpha
        pc = PatchCollection([rect], facecolor=facecolor, alpha=a,
                             edgecolor=edgecolor)

        # Add collection to axes
        ax.add_collection(pc)

    # Plot baseline
    artists = ax.plot(np.arange(0,1.1,.05), np.arange(0,1.1,.05), '--')
    return artists


def prep_accs(confidences, corrects, bins=10):
    delta = 1.0 / bins
    threshold = np.arange(0, 1 + delta, delta)
    accs = []
    freqs = []
    for idx in range(bins):
        lower_t = threshold[idx]
        upper_t = threshold[idx+1]
        idx1 = confidences > lower_t
        idx2 = confidences <= upper_t
        curr_acc = np.mean(corrects[idx1*idx2])
        curr_acc = curr_acc if not np.isnan(curr_acc) else 0
        freqs.append(np.sum(idx1*idx2))
        accs.append(curr_acc)
    return accs, freqs


def compute_ece(accs, freqs):
    n_bins = len(accs)
    delta = 1.0 / n_bins
    confs = np.arange(0, 1, delta) + delta/2
    freqs = np.array(freqs)
    return (np.abs(np.array(accs) - confs) * freqs / freqs.sum()).sum()


def plot_accs(accs, freqs, ax):
    font = { 'family': 'serif', 'color':  'black', 'weight': 'normal', 'size': 12 }

    # Create figure and axes
    n_bins = len(accs)
    freqs = np.array(freqs)
    def sm(x, T=10, n_bins=20):
        return np.clip(np.exp(x/T)/ np.sum(np.exp(x/T))*n_bins,a_min=.1, a_max=1) - 1e-5
    _ = make_error_boxes(ax, np.arange(0,1,1/n_bins), accs, width=1/n_bins, alpha=.5)

    ax.set_xlim([0,1])
    ax.set_xlabel('confidence')
    ax.set_ylim([0,1])
    ax.set_ylabel('accuracy')
    ax.text(0.1, .9, 'ECE: {0:.2f}%'.format(compute_ece(accs, freqs)*100), fontdict=font,
            ha="center", va="center",
            bbox=dict(boxstyle="round", ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8)))


def filter_opt(opt, tag):
    ret = { }

    for k,v in opt.items():
        tokens = k.split('.')
        if tokens[0] == tag:
            ret['.'.join(tokens[1:])] = v

    return ret


def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)
