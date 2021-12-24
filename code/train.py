import os
import json
import math
import argparse
import random
import logging
from collections import OrderedDict

import torch
import matplotlib.pyplot as plt
import numpy as np

from options.parse import parse, dict2str, save_opt
from utils import util_dir, util_log, util_misc, util_dist
from data import create_sampler, create_dataloader, create_dataset
from solvers import create_solver


def main():
    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option yaml file.')
    parser.add_argument('-extra', type=str, nargs='*', default=[], help='extra arguments, in form a.a=a')
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()
    opt = parse(args.opt, extra=args.extra, is_train=True)
    if args.local_rank >= 0:
        opt['distributed'] = True
        opt['rank'] = args.local_rank
    else:
        opt['distributed'] = False
        opt['rank'] = 0
    rank0 = opt['rank'] == 0

    #### loading resume state if exists
    if opt['path']['resume_state']:
        resume_state = torch.load(opt['path']['resume_state'])
        resume_path = opt['path']['resume_state']
        resume_epoch = os.path.basename(resume_path).split('.')[0]
        opt['path']['pretrained_model'] = os.path.join(opt['path']['model'], resume_epoch + '_model.pth')
    else:
        resume_state = None

    #### mkdir and loggers
    if resume_state is None and rank0:
        util_dir.mkdir_and_rename(
            opt['path']['experiment_root'], opt['time_stamp'], opt['no_check'])  # rename experiment folder if exists
        util_dir.mkdirs((path for key, path in opt['path'].items() if not key == 'experiment_root'
                         and 'pretrain_model' not in key 
                         and 'resume' not in key))
    # config loggers. Before it, the log will not work
    if opt['debug']:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    util_log.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=log_level,
                      screen=True, tofile=rank0)
    logger = logging.getLogger('base')
        
    # dist init
    if opt['distributed']:
        opt['num_rank'] = util_dist.dist_init(args.local_rank)
        # compute batchsize per gpu
        b_size_full = opt['dataset']['train']['batch_size']
        b_size = b_size_full // opt['num_rank']
        opt['dataset']['train']['batch_size'] = b_size
        if b_size_full != b_size * opt['num_rank']:
            logger.error(f"Full batch size {b_size_full} can\'t be divided by num of gpus {opt['num_rank']}!")
    util_dist.dist_barrier(opt)
    
    # tensorboard logger
    if opt['use_tb_logger'] and 'debug' not in opt['name'] and rank0:
        from tensorboardX import SummaryWriter
        opt['path']['tb_logger'] = os.path.join(opt['path']['root'], 'tb_logger', opt['name'])
        if resume_state is None:
            util_dir.mkdir_and_rename(opt['path']['tb_logger'], opt['time_stamp'], opt['no_check'])
        tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    #### random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    if rank0:
        logger.info('Random seed: {}'.format(seed))
    util_misc.set_random_seed(seed)

    #### log environs
    if rank0:
        write_str = ''
        for k, v in os.environ.items():
            tmp_str = '%s=%s' % (k, v)
            if opt['debug']:
                logger.info(tmp_str)
            write_str += tmp_str + '\n'
        with open(os.path.join(opt['path']['experiment_root'], 'environ.txt'), 'w') as f:
            f.write(write_str)

    #### create train and val dataloader
    for phase, dataset_opt in opt['dataset'].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt, opt)
            train_sampler = create_sampler(train_set, opt, shuffle=True)
            train_loader = create_dataloader(train_set, dataset_opt, train_sampler)
            train_size = len(train_loader)
            total_epochs = int(opt['train']['num_epoch'])
            if rank0:
                logger.info('Number of train samples: %d, epochs: %d' % (len(train_set), total_epochs))
        elif phase == 'val':
            pass
#             val_set = create_dataset(dataset_opt, opt)
#             val_sampler = create_sampler(val_set, opt, shuffle=False)
#             val_loader = create_dataloader(val_set, dataset_opt, val_sampler)
#             if rank0:
#                 logger.info('Number of val samples in [{:s}]: {:d}'.format(
#                     dataset_opt['name'], len(val_set)))
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
    assert train_loader is not None

    # update steps for shceduler and model
    util_misc.update_opt(opt, train_size)

    if resume_state is None:
        save_opt(os.path.join(opt['path']['experiment_root'], 'opt.yaml'), opt)
        opt['model']['start_step'] = 0
    else:
        opt['model']['start_step'] = resume_state['step']

    #### create solver
    solver = create_solver(opt)

    #### resume training
    if resume_state:
        if rank0:
            logger.info('Resuming training from epoch: {}, step: {}.'.format(
                resume_state['epoch'], resume_state['step']))
        solver.resume_training(resume_state)  # handle optimizers and schedulers


    if rank0:
        logger.info(dict2str(opt))
    
    #### training
    start_epoch = solver.epoch
    if rank0:
        logger.info('Start training from epoch: {:d}, step: {:d}'.format(start_epoch, solver.step))
    for epoch in range(start_epoch, total_epochs):
        util_dist.dist_barrier(opt)
        for _, train_data in enumerate(train_loader):

            #### training
            solver.feed_data(train_data)
            solver.optimize_parameter()

            #### log
            if solver.cur_step % opt['logger']['print_freq'] == 0:
                if opt['use_tb_logger'] and rank0:
                    solver.log_current(train_size, tb_logger)
                elif rank0:
                    solver.log_current(train_size)
            solver.update_learning_rate()
            if opt['train']['save_freq_step'] is not None and solver.step % opt['train']['save_freq_step'] == 0:
                if rank0:
                    solver.save('step_%07d' % solver.step)
                    solver.save_training_state()
        solver.new_epoch(train_sampler)
        #### save models and training states
        if opt['train']['save_freq'] is None or solver.epoch % opt['train']['save_freq'] == 0:
            if rank0:
                solver.save('%04d' % solver.epoch)
                solver.save_training_state()
            
        
    if rank0:
        with open(os.path.join(opt['path']['experiment_root'], 'final_log.json'), 'w') as f:
            json.dump(solver.log_dict, f, indent=2)

    logger.info('End of training.')
    util_dist.dist_destroy(opt)
    
if __name__ == '__main__':
    main()
