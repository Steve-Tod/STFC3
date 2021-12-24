import time
import queue
import logging
from collections import OrderedDict

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torchnet.meter import AverageValueMeter

from .BaseSolver import BaseSolver
from models.losses import *
from models import define_net
from utils.util_misc import GradualWarmupScheduler, compute_acc
from utils.LARS import LARS, add_weight_decay

logger = logging.getLogger('base')

try:
    from apex import amp
except ImportError:
    amp = None


class STFC3Solver(BaseSolver):
    # use mask in loss
    def __init__(self, opt):
        super().__init__(opt)
        model_opt = opt['model']
        self.model = define_net(model_opt, rank=self.rank).to(self.device)

        self.print_network()  # print network
        self.load()  # load model if needed

        if self.is_train:
            train_opt = opt['train']
            self.model.train()
            self.log_dict = OrderedDict()
            self.build_loss(train_opt)
            self.label_flag = True
            self.time_meter = AverageValueMeter()
            self.acc_meter = AverageValueMeter()
            self.log_dict['step_time'] = {}
            self.log_dict['acc'] = {}
            # optimizer
            if train_opt['optimizer'] == 'Adam':
                self.optimizer = torch.optim.Adam(
                    filter(lambda x: x.requires_grad, self.model.parameters()),
                    lr=train_opt['learning_rate'],
                    weight_decay=train_opt['weight_decay'])
            elif train_opt['optimizer'] == 'AdamW':
                self.optimizer = torch.optim.AdamW(
                    filter(lambda x: x.requires_grad, self.model.parameters()),
                    lr=train_opt['learning_rate'],
                    weight_decay=train_opt['weight_decay'])
            elif train_opt['optimizer'] == 'LARS':
                params = add_weight_decay(self.model, train_opt['weight_decay'])
                base_optimizer = torch.optim.SGD(
                    params,
                    lr=train_opt['learning_rate'],
                    momentum=0.9)
                self.optimizer = LARS(base_optimizer)
            else:
                raise NotImplementedError(
                    'Optimizer type %s not implemented!' %
                    train_opt['optimizer'])
            self.optimizer_list.append(self.optimizer)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizer_list:
                    scheduler = torch.optim.lr_scheduler.MultiStepLR(
                        optimizer,
                        train_opt['lr_steps'],
                        gamma=train_opt['lr_gamma'],
                        last_epoch=self.step - 1)
                    if 'num_warmup_step' in train_opt.keys():
                        scheduler = GradualWarmupScheduler(
                            optimizer, train_opt['num_warmup_step'], scheduler)
                    self.scheduler_list.append(scheduler)
            elif train_opt['lr_scheme'] == 'Cosine':
                T_max = opt['model']['total_step']
                for optimizer in self.optimizer_list:
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer,
                        T_max=T_max,
                        last_epoch=self.step - 1)
                    if 'num_warmup_step' in train_opt.keys():
                        scheduler = GradualWarmupScheduler(
                            optimizer, train_opt['num_warmup_step'], scheduler)
                    self.scheduler_list.append(scheduler)
            else:
                raise NotImplementedError(
                    'Learning rate schedule scheme %s is not implemented' %
                    train_opt['lr_scheme'])
        else:
            self.test_opt = opt['test']

        if self.amp_opt_level != 'O0':
            if self.is_train:
                self.model, self.optimizer = amp.initialize(
                    self.model, self.optimizer, opt_level=self.amp_opt_level)
            else:
                self.model = amp.initialize(self.model,
                                            opt_level=self.amp_opt_level)

        if self.distributed:
            if model_opt['sync_bn']:
                self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
                    self.model)
            self.model = DDP(self.model,
                             device_ids=[self.rank],
                             output_device=self.rank)

    def build_loss(self, train_opt):
        self.cri_dict = OrderedDict()
        for k, v in train_opt['loss'].items():
            self.cri_dict[k] = {}
            self.cri_dict[k]['weight'] = v['weight']
            if v['loss_type'] == 'BCEMaskNeighbour':
                self.cri_dict[k]['cri'] = BCEMaskNeighbourLoss(v)
            elif v['loss_type'] == 'BCEMask':
                self.cri_dict[k]['cri'] = BCEMaskLoss
            elif v['loss_type'] == 'LogNLLMaskNeighbour':
                self.cri_dict[k]['cri'] = LogNLLMaskNeighbourLoss(v)
            elif v['loss_type'] == 'NLLMask':
                self.cri_dict[k]['cri'] = NLLMaskLoss(v)
            elif v['loss_type'] == 'KLDMask':
                self.cri_dict[k]['cri'] = KLDMaskLoss
            else:
                raise NotImplementedError('Loss type %s not implemented!' %
                                          v['loss_type'])
            self.cri_dict[k]['meter'] = AverageValueMeter()
            self.log_dict['loss_' + k] = {}

    def feed_data(self, data, inference=False):
        data, affine_mat, label, mask = data
        # B*T*P*C*h*w
        self.data = data.to(self.device)
        self.affine_mat = affine_mat.to(self.device)
        self.label = label.to(self.device)
        self.mask = mask.to(self.device)

    # train one step
    def optimize_parameter(self):
        start = time.time()
        self.model.train()
        self.optimizer.zero_grad()
        At_list = self.model(self.data, self.affine_mat)

        if len(self.mask.shape) == 2:
            self.mask.unsqueeze_(1)
            self.mask = self.mask.expand(-1, len(At_list), -1)

        loss = 0.0
        acc = 0.0
        for idx, A in enumerate(At_list):
            loss += self.cri_dict['cycle']['cri'](A, self.label,
                                                  self.mask[:, idx])
            acc += compute_acc(A, self.label, self.mask[:, idx])
        loss *= self.cri_dict['cycle']['weight']
        acc /= len(At_list)
        if self.opt['train']['sub_reduce'] != 'sum':
            loss /= len(At_list)
        self.cri_dict['cycle']['meter'].add(loss.item())

        if self.amp_opt_level != "O0":
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        self.optimizer.step()
        duration = time.time() - start
        self.time_meter.add(duration)
        self.acc_meter.add(acc)
        self.step += 1
        self.cur_step += 1
        self.log_dict['acc']['cur'] = acc
        self.log_dict['acc']['mean'] = self.acc_meter.value()[0]
        self.log_dict['loss_cycle']['cur'] = loss.item()
        self.log_dict['loss_cycle']['mean'] = self.cri_dict['cycle'][
            'meter'].value()[0]
        self.log_dict['step_time']['cur'] = duration
        self.log_dict['step_time']['mean'] = self.time_meter.value()[0]

    def update_learning_rate(self):
        # TODO check scheduler
        for s in self.scheduler_list:
            s.step()

    def new_epoch(self, sampler):
        super().new_epoch(sampler)
        self.time_meter.reset()
        for k, v in self.cri_dict.items():
            v['meter'].reset()

    def log_current(self, train_size, tb_logger=None):
        logs = self.log_dict
        message = '<epoch:%3d, iter:%6d/%6d, lr:%.3e> ' % (
            self.epoch, self.cur_step, train_size,
            self.get_current_learning_rate())
        for k, v in logs.items():
            message += '%s: %.3f (%.3f); ' % (k, v['cur'], v['mean'])
            # tensorboard logger
        if self.rank == 0:
            logger.info(message)
        if tb_logger is not None:
            for k, v in logs.items():
                tb_logger.add_scalar('loss/%s' % k, v['cur'], self.step)

    def load(self):
        if self.opt['path']['strict_load'] is None:
            strict = True
        else:
            strict = self.opt['path']['strict_load']
        load_path = self.opt['path']['pretrained_model']
        if load_path is not None:
            logger.info('[Rank %d] Loading model from %s ...' %
                        (self.rank, load_path))
            self.load_network(load_path, self.model, strict)

    def save(self, save_label):
        self.save_network(self.model, 'model', save_label)
