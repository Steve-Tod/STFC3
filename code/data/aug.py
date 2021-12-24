from torchvision import transforms
from PIL import ImageFilter, ImageOps
import kornia
import kornia.augmentation as K
import torch
import numpy as np

from .RandomResizedCropCustom import RandomResizedCropCustom

class RandomResizedCropHFilp(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.size = opt['size']
        if 'power' in opt.keys():
            self.rrcrop = RandomResizedCropCustom(size=(self.size, self.size),
                                            scale=opt['aug_scale'],
                                            ratio=opt['aug_ratio'],
                                            same_on_batch=opt['same_on_batch'],
                                            power=opt['power'],
                                            return_transform=True)
        else:
            self.rrcrop = K.RandomResizedCrop(size=(self.size, self.size),
                                            scale=opt['aug_scale'],
                                            ratio=opt['aug_ratio'],
                                            same_on_batch=opt['same_on_batch'],
                                            return_transform=True)
        if opt['h_flip']:
            self.hflip = K.RandomHorizontalFlip(
                p=opt['h_flip'],
                same_on_batch=opt['same_on_batch'],
                return_transform=True)
        else:
            self.hflip = None

    def forward(self, x):
        B, C, H, W = x.shape
        x, transmat = self.rrcrop(x)
        transmat = kornia.geometry.warp.normalize_homography(
            transmat, (H, W), (self.size, self.size))
        if self.hflip is not None:
            x, transmat_ = self.hflip(x)
            transmat_[:, 0, 2] = 0
            transmat = torch.bmm(transmat_, transmat)
        return x, transmat


class GaussianBlur(object):
    """Gaussian Blur version 2"""

    def __call__(self, x):
        sigma = np.random.uniform(0.1, 2.0)
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def get_transform(aug_type):
    if aug_type == 'MoCov2':  # used in MoCov2
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur()], p=0.5),
            transforms.ToTensor(),
        ])
    elif aug_type == 'SimCLR':  # used in SimCLR and PIC
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur()], p=0.5),
            transforms.ToTensor(),
        ])
    elif aug_type == 'BYOL':
        transform_1 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur()], p=1.0),
            transforms.ToTensor(),
        ])
        transform_2 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur()], p=0.1),
            transforms.RandomApply([ImageOps.solarize], p=0.2),
            transforms.ToTensor(),
        ])
        transform = (transform_1, transform_2)
    elif aug_type is None:
        transform = None
    else:
        supported = '[MoCov2, SimCLR, BYOL]'
        raise NotImplementedError(f'aug_type "{aug_type}" not supported. Should in {supported}')

    return transform