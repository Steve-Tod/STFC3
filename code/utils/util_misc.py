import random
import numpy as np
import torch

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def update_opt(opt, train_size):
    train_opt = opt['train']
    if train_opt['lr_scheme'] == 'MultiStepLR':
        train_opt['lr_steps'] = [x * train_size for x in train_opt['lr_steps']]
    elif train_opt['lr_scheme'] == 'LinearWarmup':
        if 'num_warmup_step' not in train_opt.keys():
            train_opt['num_warmup_step'] = train_opt['num_warmup_epoch'] * train_size
        train_opt['num_step'] = train_opt['num_epoch'] * train_size
    opt['model']['total_step'] = train_opt['num_epoch'] * train_size

def compute_acc(prob, label, mask=None):
    '''
    prob: B, P, P, [0, 1]
    label: B, P, P, binary
    mask: B, P
    '''
    # B, P
    with torch.no_grad():
        hit = prob.argmax(-1) == label.argmax(-1)
        hit = hit.float()
        if mask is None:
            acc = hit.mean()
        else:
            hit *= mask
            acc = hit.sum() / (mask.sum() + 1e-5)
        return acc.item()

class GradualWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, total_epoch, after_scheduler):
        self.multiplier = 1
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]


    def step(self):
        self.after_scheduler.step()
        super().step()
