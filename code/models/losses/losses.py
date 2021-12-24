import torch
from torch import nn
import torch.nn.functional as F

def log(x, eps=1e-7):
    return torch.log(x + eps)

class LogNLLLoss(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.nll = nn.NLLLoss(ignore_index=-1)
    def forward(self, x, label):
        if x.min().item() < 0:
            print('min val %.2e < 0!' % x.min().item())
        if x.max().item() > 1.1:
            print('max val %.2e > 1!' % x.max().item())
        x = torch.log(x.transpose(-1, -2) + self.eps)
        return self.nll(x, label)

class LogNLLNeighbourLoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.reduction = opt['reduction']
        self.aggregation = opt['aggregation']

    def forward(self, output, label):
        '''
        output: B, N1, N2
        label: B, N1, N2, binary
        '''
        if self.aggregation == 'sum':
            pos_prob = (output * label).sum(-1)
            loss = log(pos_prob)
        elif self.aggregation is None:
            # no aggregation, mean of log prob
            loss = (log(output) * label).sum(-1)
            loss /= label.sum(-1)
        loss *= -1

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss

class LogCrossEntropyLoss(nn.Module):
    def __init__(self, eps=1e-20):
        super().__init__()
        self.eps = eps
        self.ce = nn.CrossEntropyLoss(ignore_index=-1)
    def forward(self, x, label):
        if x.min().item() < 0:
            print('min val %.2e < 0!' % x.min().item())
        if x.max().item() > 1.1:
            print('max val %.2e > 1.1!' % x.max().item())
        # original dimension is -1
        x = torch.log(x.transpose(-1, -2) + self.eps)
        return self.ce(x, label)

class NLLMaskLoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.autofill = opt['autofill']

    def forward(self, output, label, mask):
        # output: B, N, N(N')
        # label: B, N, N
        # mask: B, N
        if self.autofill and output.size(2) != label.size(2):
            label = F.pad(label, (0, output.size(2) - label.size(2)), 'constant', 0)
        loss = (output * label).sum(-1) / (label.sum(-1) + 1e-6)
        loss *= mask
        loss = -loss.sum() / (mask.sum() + 1e-5)
        return loss

def BCEMaskLoss(output, label, mask=None):
    loss = (log(output)* label).sum(-1) + (log(1 - output) * (1 - label)).sum(-1)
    if mask is not None:
        loss *= mask
        loss = -loss.sum() / (mask.sum() + 1e-7)
    return loss

def KLDMaskLoss(x, label, mask):
    loss = label * (log(label) - log(x))
    loss = loss.sum(-1)
    loss *= mask
    loss = loss.sum() / (mask.sum() + 1e-7)
    return loss

class BCEMaskNeighbourLoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        if 'aggregation' in opt.keys():
            self.aggregation = opt['aggregation']
        else:
            self.aggregation = 'sum'

    def forward(self, output, label, mask=None):
        '''
        output: B, N1, N2
        label: B, N1, N2, binary
        mask: B, N1
        '''
        # aggregate neighbor
        if self.aggregation == 'sum':
            pos_prob = (output * label).sum(-1)
        elif self.aggregation == 'max':
            pos_prob = (output * label).max(-1)[0]
        # compute loss
        loss = log(pos_prob) + (log(1 - output) * (1 - label)).sum(-1)
        # normalize by number of neg samples + 1
        # num_sample = (1 - label).sum(-1) + 1
        # loss /= num_sample
        if mask is not None:
            # mask unmatched items
            loss *= mask
            loss = -loss.sum() / (mask.sum()+1e-5)
            return loss


class LogNLLMaskNeighbourLoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.aggregation = opt['aggregation']
        self.autofill = opt['autofill']

    def forward(self, output, label, mask=None):
        '''
        output: B, N1, N2
        label: B, N1, N2, binary
        mask: B, N1
        '''
        if self.autofill and output.size(2) != label.size(2):
            label = F.pad(label, (0, output.size(2) - label.size(2)), 'constant', 0)
        # aggregate neighbor
        if self.aggregation == 'sum':
            pos_prob = (output * label).sum(-1)
            loss = log(pos_prob)
        elif self.aggregation is None:
            # no aggregation, mean of log prob
            loss = (log(output) * label).sum(-1)
            loss /= label.sum(-1)
        if mask is not None:
            # mask unmatched items
            loss *= mask
            loss = -loss.sum() / (mask.sum() + 1e-5)
        return loss
