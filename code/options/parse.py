import os
import glob
import os.path as osp
import logging
import yaml
from utils.util_yaml import OrderedYaml
from utils.util_dir import get_timestamp
Loader, Dumper = OrderedYaml()

def parse(opt_path, extra=[], is_train=True):
    with open(opt_path, 'r') as f:
        opt = yaml.load(f, Loader=Loader)

    opt['time_stamp'] = get_timestamp()
    opt['is_train'] = is_train

    parse_extra(opt, extra)
    # path
    if is_train:
        experiment_root = osp.join(opt['path']['root'], 'experiments', opt['name'])
        opt['path']['experiment_root'] = experiment_root
        opt['path']['model'] = osp.join(experiment_root, 'model')
        opt['path']['training_state'] = osp.join(experiment_root, 'training_state')
        opt['path']['log'] = experiment_root
        opt['path']['visual'] = osp.join(experiment_root, 'visual')
    else:  # test
        result_root = osp.join(opt['path']['root'], 'results', opt['name'])
        opt['path']['result_root'] = result_root
        opt['path']['log'] = result_root
    
    if 'gpu_id' in opt.keys():
        gpu_list = ','.join(str(x) for x in opt['gpu_id'])
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
        print('export CUDA_VISIBLE_DEVICES=' + gpu_list)

    return dict_to_nonedict(opt)

def parse_extra(opt, extra):
    for kv in extra:
        k, v = kv.split('=')
        tmp_dict = opt
        ks = k.split('.')
        for sub_k in ks[:-1]:
            tmp_dict = tmp_dict[sub_k]
        v = yaml.load('k: %s' % (v), Loader=Loader)['k']
        tmp_dict[ks[-1]] = v
    return opt

class NoneDict(dict):
    '''
    return None for missing key
    '''
    def __missing__(self, key):
        return None

def save_opt(path, opt):
    if opt['rank'] == 0:
        with open(path, 'w') as f:
            yaml.dump(opt, f, Dumper=Dumper)

def save_train_opt(pretrain_path, save_path):
    if 'experiment' in pretrain_path:
        train_exp_path = '/'.join(pretrain_path.split('/')[:-2])
        train_opt_path = os.path.join(train_exp_path, 'opt.yaml')
        assert os.path.isfile(train_opt_path), train_opt_path
        print('Saving train opt %s' % train_opt_path)
        os.system('cp %s %s' % (train_opt_path, save_path))

# convert to NoneDict, which return None for missing key.
def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt
    
def dict2str(opt, indent_l=1):
    '''dict to string for logger'''
    msg = ''
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_l * 2) + k + ':[\n'
            msg += dict2str(v, indent_l + 1)
            msg += ' ' * (indent_l * 2) + ']\n'
        else:
            msg += ' ' * (indent_l * 2) + k + ': ' + str(v) + '\n'
    return msg
