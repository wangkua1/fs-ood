import torch
import ipdb

class BaseConfidence:
    def __init__(self, f):
        self.f = f
    def support(self,s):
        pass
    def score(self, x):
        return self.f(x)

# Acquisition functions
def f_spp(x, bias=None):
    ret = torch.max(torch.exp(x), dim=-1)[0]
    # ret = torch.max(F.softmax(x, dim=-1),dim=-1)[0]
    if bias is not None:
        ret += bias.type_as(ret)
    return ret

class FSCConfidence:
    def __init__(self, fsc, f_name):
        self.fsc = fsc
        self.f_name = f_name

    def support(self,s):
        self.s = s

    def score(self, x):
        xq = x.unsqueeze(0)
        if self.f_name == 'spp':
            log_p_y = self.fsc.log_p_y(self.s, xq)['log_p_y']
            log_p_y = torch.reshape(log_p_y, [-1, log_p_y.shape[-1]])
            return f_spp(log_p_y)
            
        elif self.f_name == 'ed':
            dists = -self.fsc.log_p_y(self.s, xq)['logits'].view(-1,self.s.size(0))
            return -torch.min(dists, -1)[0]

    def eval(self):
        self.fsc.eval()

        