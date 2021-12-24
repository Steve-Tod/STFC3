import os
import logging
import numpy as np

import torch
from torchvision import transforms
import kornia.augmentation as K

from data.aug import RandomResizedCropHFilp, get_transform
from data.aug_official import get_train_transforms, IMG_MEAN, IMG_STD
from data.kinetics import Kinetics400
from data.utils import *

logger = logging.getLogger('base')

class KineticsTVDataset(torch.utils.data.Dataset):
    # for CRWFCN, return a sequence
    # add color augmentation
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.data_dir = opt['root']
        self.is_train = opt['phase'] == 'train'
        self.seq_len = opt['seq_len']
        self.frame_rate = opt['frame_rate']
        self.clips_per_video = opt['clips_per_video']
        self.feature_size = opt['feature_size']
        self.affine_th = opt['affine_th']
        self.frame_size = opt['frame_size']

        self.to_image = transforms.ToPILImage()
        opt['same_on_batch'] = True
        self.aug = RandomResizedCropHFilp(opt)
        print(self.aug)
        self.color_aug = get_transform(opt['color_aug'])
        print(self.color_aug)

        self.normalize = K.Normalize(mean=torch.Tensor(IMG_MEAN), std=torch.Tensor(IMG_STD))
        self.denormalize = K.Denormalize(mean=torch.Tensor(IMG_MEAN), std=torch.Tensor(IMG_STD))

        self.feature_mask = torch.ones(1, 1, self.feature_size,
                                       self.feature_size)

        assert self.is_train
        # no norm
        transform_train = get_train_transforms(opt, norm=False)
        if opt['metadata'] is not None:
            dataset, _ = torch.load(opt['metadata'])
            if dataset.video_clips.video_paths[0].startswith(self.data_dir):
                video_paths = dataset.video_clips.video_paths
            else:
                video_paths = []
                for p in dataset.video_clips.video_paths:
                    new_p = '/'.join(p.split('/')[-3:])
                    new_p = os.path.join(self.data_dir, new_p)
                    video_paths.append(new_p)
            cached = dict(video_paths=video_paths,
                    video_fps=dataset.video_clips.video_fps,
                    video_pts=dataset.video_clips.video_pts)
        else:
            cached = None
            
        self.Kinetics400 = Kinetics400(
            self.data_dir,
            step_between_clips=1,
            frames_per_clip=self.seq_len,
            frame_rate=self.frame_rate,
            extensions=('mp4'),
            transform=transform_train,
            _precomputed_metadata=cached)

        if opt['label_size'] is None:
            label_size = self.feature_size
        else:
            label_size = opt['label_size']
        self.label = generate_neighbour_label(label_size,
                                              label_size,
                                              opt['dist_type'],
                                              opt['rad']).float()

    def __getitem__(self, idx, return_orig=False):
        data = self.Kinetics400[idx]
        x, orig = data[0]
        x = torch.from_numpy(x).float() / 255.0
        x = x.permute(0, 3, 1, 2)
        T, C, H, W = x.shape

        assert C == 3

        # color augmentation
        x, x_backward = self.augment(x)

        frames_forward, affine_mat_forward = self.aug(x)
        frames_backward, affine_mat_backward = self.aug(x_backward)


        frames = torch.cat((frames_forward, frames_backward), dim=0)
        frames = self.normalize(frames)

        affine_mat_backward_inv = torch.inverse(affine_mat_backward[0])
        affine_mat_f2b = torch.matmul(affine_mat_forward[0], affine_mat_backward_inv)
        affine_mat_f2b = affine_mat_f2b[:2]
        
        affine_grid = torch.nn.functional.affine_grid(
            theta=affine_mat_f2b.unsqueeze(0),
            size=(1, 1, self.feature_size, self.feature_size),
            align_corners=True)
        feature_mask = torch.nn.functional.grid_sample(self.feature_mask,
                                                       affine_grid,
                                                       align_corners=True)
        feature_mask_b = feature_mask > self.affine_th
        feature_mask_b = feature_mask_b.view(-1).float()

        if return_orig:
            return frames, affine_mat_f2b, self.label, feature_mask_b, orig
        else:
            return frames, affine_mat_f2b, self.label, feature_mask_b

    def augment(self, frames):
        frames_aug = []
        frames_aug_back = []
        if isinstance(self.color_aug, tuple):
            for t in range(frames.size(0)):
                frames_aug.append(self.color_aug[0](frames[t]))
            for t in reversed(range(frames.size(0))):
                frames_aug_back.append(self.color_aug[1](frames[t]))
        elif self.color_aug is None:
            return frames, frames.flip(0)
        else:
            for t in range(frames.size(0)):
                frames_aug.append(self.color_aug(frames[t]))
            for t in reversed(range(frames.size(0))):
                frames_aug_back.append(self.color_aug(frames[t]))
        return torch.stack(frames_aug, dim=0), torch.stack(frames_aug_back, dim=0)
        
    def detrans(self, tensor):
        img_list = []
        for i in range(tensor.size(0)):
            img_tensor = tensor[i]  # 3, 256, 256
            mean = torch.Tensor(IMG_MEAN)
            mean = mean.float().view(3, 1, 1)
            std = torch.Tensor(IMG_STD)
            std = std.float().view(3, 1, 1)
            img_tensor = img_tensor * std + mean
            img_list.append(self.to_image(img_tensor))
        return img_list

    def __len__(self):
        return len(self.Kinetics400)
