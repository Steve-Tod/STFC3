import torch
import torch.utils.data
import logging
from torchvision.datasets.samplers.clip_sampler import DistributedSampler, RandomClipSampler
from torch.utils.data.dataloader import default_collate

def collate_fn_random_scale(batch):
    num_scale = len(batch[0])
    select_scale = torch.randint(num_scale, size=(1,)).item()
    return default_collate([d[select_scale] for d in batch])

def create_sampler(dataset, opt, shuffle):
    no_sampler = True
    if hasattr(dataset, 'Kinetics400'):
        sampler = RandomClipSampler(dataset.Kinetics400.video_clips, dataset.clips_per_video)
        dataset = sampler
        no_sampler = False
    if opt['distributed'] and not opt['no_sampler']:
        sampler = DistributedSampler(dataset, rank=opt['rank'], shuffle=shuffle)
        no_sampler = False
    if no_sampler:
        sampler = None
    return sampler

def create_dataloader(dataset, dataset_opt, sampler=None):
    phase = dataset_opt['phase']

    if dataset.__class__.__name__ in ['KineticsTVDatasetV8']:
        collate_fn = collate_fn_random_scale
    else:
        collate_fn = default_collate
    
    if phase == 'train':
        num_workers = dataset_opt['num_workers']
        batch_size = dataset_opt['batch_size']
        shuffle = (sampler is None)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                           num_workers=num_workers, sampler=sampler, collate_fn=collate_fn, drop_last=True,
                                           pin_memory=True)
    else:
        return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, sampler=sampler, collate_fn=collate_fn, pin_memory=True)

def create_dataset(dataset_opt, opt):
    mode = dataset_opt['mode']
    if mode == 'KineticsTVDataset':
        from .KineticsTVDataset import KineticsTVDataset as d
    else:
        raise NotImplementedError('Dataset [{:s}] is not recognized.'.format(mode))
        
    if 'Cache' in mode:
        if opt['distributed']:
            opt['no_sampler'] = True
            dataset_opt['part'] = (opt['rank'], opt['num_rank'])
    dataset = d(dataset_opt)
    if opt['rank'] == 0:
        logger = logging.getLogger('base')
        logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                               dataset_opt['name']))
    return dataset
