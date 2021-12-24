import logging
import os
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

logger = logging.getLogger('base')

try:
    from apex import amp
except ImportError:
    amp = None

class BaseSolver:
    def __init__(self, opt):
        self.opt = opt
        self.rank = opt['rank']
        self.distributed = opt['distributed']
        if not torch.cuda.is_available():
            self.device = torch.device('cpu')
        elif self.distributed:
            torch.cuda.set_device(self.rank)
            #self.device = torch.device(self.rank)
            self.device = torch.device('cuda')
        elif opt['gpu_id'] is not None:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        self.amp_opt_level = opt['amp_opt_level']
        if self.amp_opt_level == None:
            self.amp_opt_level = 'O0'

        if self.amp_opt_level != 'O0':
            try:
                from apex import amp
            except ImportError:
                self.amp_opt_level = 'O0'
                if self.rank == 0:
                    logger.warning('Import amp from apex failed, not using amp')
        
        if self.rank > 0:
            assert self.distributed, f'Rank {self.rank} but not distributed!'
        self.is_train = opt['is_train']
        self.checkpoint = opt['checkpoint']
        self.scheduler_list = []
        self.optimizer_list = []
        self.epoch = 0
        self.step = 0
        self.cur_step = 0
        self.best_epoch = 0
        self.best_metric = 1e5
    
    def build_loss(self, opt):
        pass
    
    def feed_data(self, data):
        pass
    
    def optimize_parameter(self):
        pass
    
    def print_network(self):
        pass
    
    def save(self, label):
        pass
    
    def load(self):
        pass
    
    def new_epoch(self, sampler):
        self.epoch += 1
        self.cur_step = 0
        if sampler is not None:
            if hasattr(sampler, 'set_epoch'):
                sampler.set_epoch(self.epoch)
        
    def update_learning_rate(self):
        for s in self.scheduler_list:
            s.step()
            
    def get_current_learning_rate(self):
        return self.optimizer_list[0].param_groups[0]['lr']
    
    def print_network(self):
        # Generator
        s, n = self.get_network_description(self.model)
        if isinstance(self.model, nn.DataParallel):
            net_struc_str = '{} - {}'.format(
                self.model.__class__.__name__,
                self.model.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.model.__class__.__name__)
        if self.rank == 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(
                net_struc_str, n))
            logger.info(s)
        
    def get_network_description(self, network):
        '''Get the string and total parameters of the network'''
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        s = str(network)
        n = sum(map(lambda x: x.numel(), network.parameters()))
        return s, n

    def save_network(self, network, network_label, epoch_label):
        save_filename = '%s_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.opt['path']['model'], save_filename)
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        logger.info('Saving model to %s' % save_path)
        torch.save(state_dict, save_path)

    def load_network(self, load_path, network, strict=True):
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        load_net = torch.load(load_path)
        load_net_clean = OrderedDict()  
        # remove unnecessary 'module.'
        for k, v in load_net.items():
            if k.startswith('module.'):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v
        network.load_state_dict(load_net_clean, strict=strict)

    def save_training_state(self):
        '''Saves training state during training, which will be used for resuming'''
        state = {'epoch': self.epoch,
                 'step': self.step,
                 'best_metric': self.best_metric,
                 'schedulers': [], 
                 'optimizers': []}
        if self.amp_opt_level != "O0":
            state['amp'] = amp.state_dict()
        for s in self.scheduler_list:
            state['schedulers'].append(s.state_dict())
        for o in self.optimizer_list:
            state['optimizers'].append(o.state_dict())
            
        save_filename = '%04d.state' % (self.epoch)
        save_path = os.path.join(self.opt['path']['training_state'], save_filename)
        logger.info('Saving training state to %s' % save_path)
        torch.save(state, save_path)

    def resume_training(self, resume_state):
        '''Resume the optimizers and schedulers for training'''
        resume_optimizer_list = resume_state['optimizers']
        resume_scheduler_list = resume_state['schedulers']
        if self.amp_opt_level != "O0" and 'amp' in resume_state.keys():
            amp.load_state_dict(resume_state['amp'])
        assert len(resume_optimizer_list) == len(self.optimizer_list), 'Wrong lengths of optimizers'
        assert len(resume_scheduler_list) == len(self.scheduler_list), 'Wrong lengths of schedulers'
        for i, o in enumerate(resume_optimizer_list):
            self.optimizer_list[i].load_state_dict(o)
        for i, s in enumerate(resume_scheduler_list):
            self.scheduler_list[i].load_state_dict(s)
        if 'best_metric' in resume_state.keys():
            self.best_metric = resume_state['best_metric']
        self.epoch = resume_state['epoch']
        self.step = resume_state['step']
