# copied from: https://github.com/wyharveychen/CloserLookFewShot/blob/master/backbone.py
# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate
import ipdb
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm

def init_layer(L):
    # Initialization using fan-in
    if isinstance(L, nn.Conv2d):
        n = L.kernel_size[0] * L.kernel_size[1] * L.out_channels
        L.weight.data.normal_(0, math.sqrt(2.0 / float(n)))
    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)

class distLinear(nn.Module):
    def __init__(self, indim, outdim):
        super(distLinear, self).__init__()
        self.L = nn.Linear( indim, outdim, bias = False)
        self.class_wise_learnable_norm = True  #See the issue#4&8 in the github 
        if self.class_wise_learnable_norm:      
            WeightNorm.apply(self.L, 'weight', dim=0) #split the weight update component to direction and norm      

        if outdim <=200:
            self.scale_factor = 2; #a fixed scale factor to scale the output of cos value into a reasonably large input for softmax, for to reproduce the result of CUB with ResNet10, use 4. see the issue#31 in the github 
        else:
            self.scale_factor = 10; #in omniglot, a larger scale factor is required to handle >1000 output classes.

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim =1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm+ 0.00001)
        if not self.class_wise_learnable_norm:
            L_norm = torch.norm(self.L.weight.data, p=2, dim =1).unsqueeze(1).expand_as(self.L.weight.data)
            self.L.weight.data = self.L.weight.data.div(L_norm + 0.00001)
        cos_dist = self.L(x_normalized) #matrix product by forward function, but when using WeightNorm, this also multiply the cosine distance by a class-wise learnable norm, see the issue#4&8 in the github
        scores = self.scale_factor* (cos_dist) 

        return scores


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class Conv2d_fw(nn.Conv2d): # Used in MAML to forward input with fast weight
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,padding=0, bias = True):
        super(Conv2d_fw, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.weight.fast = None
        if not self.bias is None:
            self.bias.fast = None

    def forward(self, x):
        if 'fast' not in dir(self.weight):
            self.weight.fast = None
            if not self.bias is None:
                self.bias.fast = None

        if self.bias is None:
            if self.weight.fast is not None:
                out = F.conv2d(x, self.weight.fast, None, stride= self.stride, padding=self.padding)
            else:
                out = super(Conv2d_fw, self).forward(x)
        else:
            if self.weight.fast is not None and self.bias.fast is not None:
                out = F.conv2d(x, self.weight.fast, self.bias.fast, stride= self.stride, padding=self.padding)
            else:
                out = super(Conv2d_fw, self).forward(x)

        return out


class Conv2d_fw_bbb(nn.Conv2d): # Used in MAML to forward input with fast weight
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,padding=0, bias = True):
        super(Conv2d_fw_bbb, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        assert bias
        self.weight.fast = None
        self.bias.fast = None
        self.weight_std = nn.Parameter(-10*torch.ones_like(self.weight).to(self.weight.device))
        self.weight_std.fast = None
        self.bias_std = nn.Parameter(-10*torch.ones_like(self.bias).to(self.weight.device))
        self.bias_std.fast = None

    def get_sample_stats(self):
        return (
                 [self.sampled_w, self.sampled_b],
                 [self.weight, self.bias],
                 [self.weight_std, self.bias_std]
                )

    def forward(self, x):
        if self.weight.fast is not None and self.bias.fast is not None:
            self.sampled_w.fast = self.weight.fast + torch.randn_like(self.weight.fast) * F.softplus(self.weight_std.fast)
            self.sampled_b.fast = self.bias.fast + torch.randn_like(self.bias.fast) * F.softplus(self.bias_std.fast)
            out = F.conv2d(x, self.sampled_w.fast, self.sampled_b.fast, stride= self.stride, padding=self.padding)
        else:

            self.sampled_w = self.weight + torch.randn_like(self.weight) * F.softplus(self.weight_std)

            self.sampled_b = self.bias + torch.randn_like(self.bias) * F.softplus(self.bias_std)
            out = F.conv2d(x, self.sampled_w, self.sampled_b, stride= self.stride, padding=self.padding)
        return out


class BatchNorm2d_fw(nn.BatchNorm2d): #used in MAML to forward input with fast weight 
    def __init__(self, num_features):
        super(BatchNorm2d_fw, self).__init__(num_features)
        self.weight.fast = None
        self.bias.fast = None

    def forward(self, x):
        if 'fast' not in dir(self.weight):
            self.weight.fast = None
            if not self.bias is None:
                self.bias.fast = None
        running_mean = torch.zeros(x.data.size()[1]).type_as(x)
        running_var = torch.ones(x.data.size()[1]).type_as(x)
        
        if self.training: 
            if self.weight.fast is not None and self.bias.fast is not None:
                # print("[Learner: ] updating")
                out = F.batch_norm(x, running_mean, running_var, self.weight.fast, self.bias.fast, training = True, momentum = 1)
                #batch_norm momentum hack: follow hack of Kate Rakelly in pytorch-maml/src/layers.py
            else:
                # print("[Learner: ] 1st step")
                out = F.batch_norm(x, running_mean, running_var, self.weight, self.bias, training = True, momentum = 1)
            self.running_var = running_var
            self.running_mean = running_mean
        else: # this basically is used after "inner-loop" is done on the support set, and we're "evaluating" on the support set
            # print("[Learner: ] predicting")
            # the line below would just work for metabn.. 
            out = F.batch_norm(x, self.running_mean,self.running_var, self.weight.fast, self.bias.fast, training = False, momentum = 0)
        return out



class BBBConvBlock(nn.Module):
    def __init__(self, indim, outdim, pool = True, padding = 1, relu=True):
        super(BBBConvBlock, self).__init__()
        self.indim  = indim
        self.outdim = outdim
        self.C = Conv2d_fw_bbb(indim, outdim, 3, padding = padding)
        self.BN = BatchNorm2d_fw(outdim)
        self.relu = nn.ReLU(inplace=True)

        # WARNING: having BN here pretty much means this is not Bayesian..
        self.parametrized_layers = [self.C, self.BN]
        if relu:
            self.parametrized_layers += [self.relu]

        if pool:
            self.pool   = nn.MaxPool2d(2)
            self.parametrized_layers.append(self.pool)

        for layer in self.parametrized_layers:
            init_layer(layer)

        self.trunk = nn.Sequential(*self.parametrized_layers)

    def get_sample_stats(self):
        return self.C.get_sample_stats()

    def forward(self,x):
        out = self.trunk(x)
        return out
        
class Linear_fw(nn.Linear): #used in MAML to forward input with fast weight 
    def __init__(self, in_features, out_features):
        super(Linear_fw, self).__init__(in_features, out_features)
        self.weight.fast = None #Lazy hack to add fast weight link
        self.bias.fast = None

    def forward(self, x):
        if 'fast' not in dir(self.weight):
            self.weight.fast = None
            if not self.bias is None:
                self.bias.fast = None
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.linear(x, self.weight.fast, self.bias.fast)
        else:
            out = super(Linear_fw, self).forward(x)
        return out


# Simple Conv Block
class ConvBlock(nn.Module):
    maml = False  # Default
    def __init__(self, indim, outdim, pool = True, padding = 1, relu=True):
        super(ConvBlock, self).__init__()
        self.indim  = indim
        self.outdim = outdim
        if self.maml:
            self.C = Conv2d_fw(indim, outdim, 3, padding = padding)
            self.BN = BatchNorm2d_fw(outdim)
        else:
            self.C = nn.Conv2d(indim, outdim, 3, padding= padding)
            self.BN = nn.BatchNorm2d(outdim)
        self.relu = nn.ReLU(inplace=True)

        self.parametrized_layers = [self.C, self.BN]
        if relu:
            self.parametrized_layers += [self.relu]

        if pool:
            self.pool   = nn.MaxPool2d(2)
            self.parametrized_layers.append(self.pool)

        for layer in self.parametrized_layers:
            init_layer(layer)

        self.trunk = nn.Sequential(*self.parametrized_layers)



    def forward(self,x):
        out = self.trunk(x)
        return out


class ConvNet(nn.Module):
    def __init__(self, depth, imgSize, flatten=True, idim=3, bbb=False):
        super(ConvNet,self).__init__()
        self.depth = depth
        trunk = []
        for i in range(depth):
            indim = idim if i == 0 else 64
            outdim = 64
            if bbb:
                B = BBBConvBlock(indim, outdim, pool=(i < 4))
            else:
                B = ConvBlock(indim, outdim, pool=(i < 4))  # Only pooling for fist 4 layers
            trunk.append(B)

        self._trunk = trunk
        self.flatten = Flatten() if flatten else lambda x:x
        self.trunk = nn.Sequential(*trunk)
        
        x = torch.zeros((2,idim,imgSize, imgSize))
        z = self.trunk(x)
        self.final_feat_dim = z.shape[1:].numel()

    def get_sample_stats(self):
        ret = [[],[],[]]
        for block in self._trunk:
            if not isinstance(block, BBBConvBlock):
                continue
            tmp = block.get_sample_stats()
            ret[0] += tmp[0]
            ret[1] += tmp[1]
            ret[2] += tmp[2]
        return ret

    def forward(self, x, no_grad=False):
        with torch.set_grad_enabled(not no_grad):
            out = self.flatten(self.trunk(x))
        return out

    def intermediate_forward(self, x, layer_index, avg_pool=False):
        assert layer_index+1 <= self.depth
        for block in self._trunk[:layer_index+1]:
            x = block(x)
        if avg_pool:
            x = torch.mean(x,[2,3])            
        return x


### ResNet ###
class ResNet(nn.Module):
    maml = False #Default
    def __init__(self,nc, block,list_of_num_layers, list_of_out_dims, flatten = True,final_fmap_size=7, first_kernel_size=7):
        # list_of_num_layers specifies number of layers in each stage
        # list_of_out_dims specifies number of output channel for each stage
        super(ResNet,self).__init__()
        assert len(list_of_num_layers)==4, 'Can have only four stages'
        self.depth = len(list_of_num_layers) + 1
        if self.maml:
            conv1 = Conv2d_fw(nc, 64, kernel_size=first_kernel_size, stride=2, padding=3,
                                               bias=False)
            bn1 = BatchNorm2d_fw(64)
        else:
            conv1 = nn.Conv2d(nc, 64, kernel_size=first_kernel_size, stride=2, padding=3,
                                               bias=False)
            bn1 = nn.BatchNorm2d(64)

        relu = nn.ReLU()
        pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        init_layer(conv1)
        init_layer(bn1)


        trunk = [conv1, bn1, relu, pool1]
        layer0 = nn.Sequential(*trunk)
        self.layers = [layer0]
        indim = 64
        for i in range(4):
            for j in range(list_of_num_layers[i]):
                half_res = (i>=1) and (j==0)
                B = block(indim, list_of_out_dims[i], half_res)
                trunk.append(B)
                self.layers.append(B)
                indim = list_of_out_dims[i]


        if flatten:
            avgpool = nn.AvgPool2d(final_fmap_size)
            trunk.append(avgpool)
            trunk.append(Flatten())
            self.final_feat_dim = indim
        else:
            self.final_feat_dim = [ indim, final_fmap_size, final_fmap_size]

        self.trunk = nn.Sequential(*trunk)
        assert len(self.layers) == self.depth

    def forward(self,x, no_grad=False):
        with torch.set_grad_enabled(not no_grad):
            out = self.trunk(x)
        return out

    def intermediate_forward(self, x, layer_index, avg_pool=False):
        assert layer_index+1 <= self.depth
        for block in self.layers[:layer_index+1]:
            x = block(x)
        if avg_pool:
            x = torch.mean(x,[2,3])            
        return x


# Simple ResNet Block
class SimpleBlock(nn.Module):
    maml = False #Default
    def __init__(self, indim, outdim, half_res):
        super(SimpleBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        if self.maml:
            self.C1 = Conv2d_fw(indim, outdim, kernel_size=3, stride=2 if half_res else 1, padding=1, bias=False)
            self.BN1 = BatchNorm2d_fw(outdim)
            self.C2 = Conv2d_fw(outdim, outdim,kernel_size=3, padding=1,bias=False)
            self.BN2 = BatchNorm2d_fw(outdim)
        else:
            self.C1 = nn.Conv2d(indim, outdim, kernel_size=3, stride=2 if half_res else 1, padding=1, bias=False)
            self.BN1 = nn.BatchNorm2d(outdim)
            self.C2 = nn.Conv2d(outdim, outdim,kernel_size=3, padding=1,bias=False)
            self.BN2 = nn.BatchNorm2d(outdim)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

        self.parametrized_layers = [self.C1, self.C2, self.BN1, self.BN2]

        self.half_res = half_res

        # if the input number of channels is not equal to the output, then need a 1x1 convolution
        if indim!=outdim:
            if self.maml:
                self.shortcut = Conv2d_fw(indim, outdim, 1, 2 if half_res else 1, bias=False)
                self.BNshortcut = BatchNorm2d_fw(outdim)
            else:
                self.shortcut = nn.Conv2d(indim, outdim, 1, 2 if half_res else 1, bias=False)
                self.BNshortcut = nn.BatchNorm2d(outdim)

            self.parametrized_layers.append(self.shortcut)
            self.parametrized_layers.append(self.BNshortcut)
            self.shortcut_type = '1x1'
        else:
            self.shortcut_type = 'identity'

        for layer in self.parametrized_layers:
            init_layer(layer)

    def forward(self, x):
        out = self.C1(x)
        out = self.BN1(out)
        out = self.relu1(out)
        out = self.C2(out)
        out = self.BN2(out)
        short_out = x if self.shortcut_type == 'identity' else self.BNshortcut(self.shortcut(x))
        out = out + short_out
        out = self.relu2(out)
        return out



# Bottleneck block
class BottleneckBlock(nn.Module):
    maml = False #Default
    def __init__(self, indim, outdim, half_res):
        super(BottleneckBlock, self).__init__()
        bottleneckdim = int(outdim/4)
        self.indim = indim
        self.outdim = outdim
        if self.maml:
            self.C1 = Conv2d_fw(indim, bottleneckdim, kernel_size=1,  bias=False)
            self.BN1 = BatchNorm2d_fw(bottleneckdim)
            self.C2 = Conv2d_fw(bottleneckdim, bottleneckdim, kernel_size=3, stride=2 if half_res else 1,padding=1)
            self.BN2 = BatchNorm2d_fw(bottleneckdim)
            self.C3 = Conv2d_fw(bottleneckdim, outdim, kernel_size=1, bias=False)
            self.BN3 = BatchNorm2d_fw(outdim)
        else:
            self.C1 = nn.Conv2d(indim, bottleneckdim, kernel_size=1,  bias=False)
            self.BN1 = nn.BatchNorm2d(bottleneckdim)
            self.C2 = nn.Conv2d(bottleneckdim, bottleneckdim, kernel_size=3, stride=2 if half_res else 1,padding=1)
            self.BN2 = nn.BatchNorm2d(bottleneckdim)
            self.C3 = nn.Conv2d(bottleneckdim, outdim, kernel_size=1, bias=False)
            self.BN3 = nn.BatchNorm2d(outdim)

        self.relu = nn.ReLU()
        self.parametrized_layers = [self.C1, self.BN1, self.C2, self.BN2, self.C3, self.BN3]
        self.half_res = half_res


        # if the input number of channels is not equal to the output, then need a 1x1 convolution
        if indim!=outdim:
            if self.maml:
                self.shortcut = Conv2d_fw(indim, outdim, 1, stride=2 if half_res else 1, bias=False)
            else:
                self.shortcut = nn.Conv2d(indim, outdim, 1, stride=2 if half_res else 1, bias=False)

            self.parametrized_layers.append(self.shortcut)
            self.shortcut_type = '1x1'
        else:
            self.shortcut_type = 'identity'

        for layer in self.parametrized_layers:
            init_layer(layer)


    def forward(self, x):

        short_out = x if self.shortcut_type == 'identity' else self.shortcut(x)
        out = self.C1(x)
        out = self.BN1(out)
        out = self.relu(out)
        out = self.C2(out)
        out = self.BN2(out)
        out = self.relu(out)
        out = self.C3(out)
        out = self.BN3(out)
        out = out + short_out

        out = self.relu(out)
        return out


def Conv4(imgSize,idim=3):
    # for Omniglot
    return ConvNet(4, imgSize, idim=idim)

def Conv4BBB(imgSize, ):
    return ConvNet(4, imgSize, bbb=True)

def ResNet10_32(nc, flatten = True):
    return ResNet(nc, SimpleBlock, [1,1,1,1],[64,128,256,512], flatten, final_fmap_size=1)

def ResNet10_84(nc, flatten = True):
    return ResNet(nc, SimpleBlock, [1,1,1,1],[64,128,256,512], flatten, final_fmap_size=2)
