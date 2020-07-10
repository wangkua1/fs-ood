# based on https://github.com/wyharveychen/CloserLookFewShot/

import backbone
import utils
import ipdb
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

class BaselineTrain(nn.Module):
    def __init__(self, model_func, num_class, loss_type = 'softmax'):
        super(BaselineTrain, self).__init__()
        self.feature    = model_func()
        if loss_type == 'softmax':
            self.classifier = nn.Linear(self.feature.final_feat_dim, num_class)
            self.classifier.bias.data.fill_(0)
        elif loss_type == 'dist': #Baseline ++
            self.classifier = backbone.distLinear(self.feature.final_feat_dim, num_class)
        self.loss_type = loss_type  #'softmax' #'dist'
        self.num_class = num_class
        self.loss_fn = nn.CrossEntropyLoss()
        self.DBval = False; #only set True for CUB dataset, see issue #31

    def forward(self,x):
        x    = Variable(x.cuda())
        out  = self.feature.forward(x)
        scores  = self.classifier.forward(out)
        return scores

    def forward_loss(self, x, y):
        scores = self.forward(x)
        y = Variable(y.cuda())
        return self.loss_fn(scores, y )
    
    def train_loop(self, epoch, train_loader, optimizer):
        print_freq = 10
        avg_loss=0

        for i, (x,y) in enumerate(train_loader):
            optimizer.zero_grad()
            loss = self.forward_loss(x, y)
            loss.backward()
            optimizer.step()

            avg_loss = avg_loss+loss.item()

            if i % print_freq==0:
                #print(optimizer.state_dict()['param_groups'][0]['lr'])
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader), avg_loss/float(i+1)  ))
                     
    def test_loop(self, val_loader):
        if self.DBval:
            return self.analysis_loop(val_loader)
        else:
            return -1   #no validation, just save model during iteration



class BaselineFinetune(nn.Module):
    def __init__(self, encoder,  n_way, n_support, loss_type = "softmax"):
        super(BaselineFinetune, self).__init__()
        self.loss_type = loss_type
        self.n_way = n_way
        self.n_support = n_support
        self.encoder = encoder
        self.feat_dim = None

    def check_feat_dim(self, xs):
        if self.feat_dim is None:
            c,h,w = xs.shape[2:]
            self.feat_dim = self.encoder(xs.view(-1,c,h,w)).shape[-1]

    def log_p_y(self,xs, xq, no_grad=False, mask=None):
        del no_grad
        del mask
        
        self.check_feat_dim(xs)
        self.encoder.eval()
        way,shot,c,h,w = xs.shape
        n_query = xq.shape[1]
        if xq.shape[0] == 1:
            n_query = n_query // way
        assert way == self.n_way and shot  == self.n_support
        with torch.no_grad():
            z_support = self.encoder(xs.view(-1, c, h, w))
            z_query = self.encoder(xq.view(-1, c, h, w))
        
        y_support = torch.from_numpy(np.repeat(range( self.n_way ), self.n_support ))
        y_support = Variable(y_support.cuda())

        if self.loss_type == 'softmax':
            linear_clf = nn.Linear(self.feat_dim, self.n_way)
        elif self.loss_type == 'dist':        
            linear_clf = backbone.distLinear(self.feat_dim, self.n_way)
        linear_clf = linear_clf.cuda()

        set_optimizer = torch.optim.SGD(linear_clf.parameters(), lr = 0.01, momentum=0.9, dampening=0.9, weight_decay=0.001)

        loss_function = nn.CrossEntropyLoss()
        loss_function = loss_function.cuda()
        
        batch_size = 4
        support_size = self.n_way* self.n_support
        for epoch in range(100):
            rand_id = np.random.permutation(support_size)
            for i in range(0, support_size , batch_size):
                set_optimizer.zero_grad()
                selected_id = torch.from_numpy( rand_id[i: min(i+batch_size, support_size) ]).cuda()
                z_batch = z_support[selected_id].detach()
                y_batch = y_support[selected_id] 
                scores = linear_clf(z_batch)
                loss = loss_function(scores,y_batch)
                loss.backward()
                set_optimizer.step()
        scores = linear_clf(z_query)

        # Ret vals
        ret_dict = {}
        ret_dict['logits']  = scores.view(way, n_query, -1)
        ret_dict['log_p_y']  = F.log_softmax(scores, -1).view(way, n_query, -1)
        n_class = way
        target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long().to(scores.device)
        ret_dict['target_inds'] = target_inds
        return ret_dict