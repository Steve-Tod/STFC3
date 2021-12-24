import os
import math
import argparse
import random
import logging
import numpy as np
from PIL import Image
from collections import OrderedDict
import matplotlib.pyplot as plt

import torch

from options.parse import parse, dict2str, save_opt
from utils import util_dir, util_log, util_misc, util_dist, util_test
from data import create_sampler, create_dataloader, create_dataset
from solvers import create_solver


def main():
    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option yaml file.')
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()
    opt = parse(args.opt, is_train=False)
    if args.local_rank >= 0:
        opt['distributed'] = True
        opt['rank'] = args.local_rank
    else:
        opt['distributed'] = False
        opt['rank'] = 0
    rank0 = opt['rank'] == 0
    assert rank0, 'Only single process testing is supported!'

    #### mkdir and loggers
    if rank0:
        util_dir.mkdir_and_rename(
            opt['path']['result_root'], opt['time_stamp'], opt['no_check'])  # rename experiment folder if exists
        util_dir.mkdirs((path for key, path in opt['path'].items() if not key == 'experiment_root'
                         and 'pretrained_model' not in key 
                         and 'resume' not in key))
    # config loggers. Before it, the log will not work
    if opt['debug']:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    util_log.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=log_level,
                      screen=True, tofile=rank0)
    logger = logging.getLogger('base')
    if rank0:
        logger.info(dict2str(opt))

    # dist init
    if opt['distributed']:
        util_dist.dist_init(args.local_rank)
    util_dist.dist_barrier(opt)

    #### create train and val dataloader
    for phase, dataset_opt in opt['dataset'].items():
        if phase == 'test':
            test_set = create_dataset(dataset_opt, opt)
            test_sampler = create_sampler(test_set, opt, shuffle=True)
            test_loader = create_dataloader(test_set, dataset_opt, test_sampler)
            test_size = len(test_loader)
            if rank0:
                logger.info('Number of test samples: %d' % (len(test_set)))
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
    assert test_loader is not None

    #### create solver
    solver = create_solver(opt)
    
    #### testing
    with torch.no_grad():
        for _, data in enumerate(test_loader):
            #### training
            frame, seg, ori_size, frame_name_list, video_name, seg_ori_path = data
            logger.info('Testing %s' % video_name[0])
            seg = seg[0][0].to(solver.device)
            # T*C*H*W
            solver.feed_data(frame.squeeze(0).cuda(), inference=True)
            feature = solver.inference()
            segs = solver.video_segmentation(feature, seg)
            save_dir = os.path.join(opt['path']['result_root'], 'seg', video_name[0])
            os.makedirs(save_dir, exist_ok=True)
            os.system('cp %s %s' % (seg_ori_path[0], os.path.join(save_dir, frame_name_list[0][0]  + '.png')))
            for i, s in enumerate(segs):
                s = torch.nn.functional.interpolate(s.unsqueeze(0),
                                        scale_factor=test_set.scale,
                                        mode='bilinear',
                                        align_corners=False)
                s = util_test.norm_mask(s[0])
                _, seg_map = torch.max(s, dim=0)
                seg_map = seg_map.cpu().numpy().astype(np.uint8)
                img = Image.fromarray(seg_map).resize(ori_size, Image.NEAREST)
                save_path = os.path.join(save_dir, frame_name_list[i + 1][0]  + '.png')
                util_test.imwrite_indexed(save_path, np.array(img))
        
    util_dist.dist_destroy(opt)
    
if __name__ == '__main__':
    main()
