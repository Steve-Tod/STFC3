import torchvision
import skimage

import torch
from torchvision import transforms

import numpy as np
from PIL import Image

IMG_MEAN = (0.4914, 0.4822, 0.4465)
IMG_STD  = (0.2023, 0.1994, 0.2010)
NORM = [transforms.ToTensor(), 
        transforms.Normalize(IMG_MEAN, IMG_STD)]


class MapTransform(object):
    def __init__(self, transforms, pil_convert=True):
        self.transforms = transforms
        self.pil_convert = pil_convert

    def __call__(self, vid):
        if isinstance(vid, Image.Image):
            return np.stack([self.transforms(vid)])
        
        if isinstance(vid, torch.Tensor):
            vid = vid.numpy()

        if self.pil_convert:
            x = np.stack([np.asarray(self.transforms(Image.fromarray(v))) for v in vid])
            return x
        else:
            return np.stack([self.transforms(v) for v in vid])
    
def n_patches(x, n, transform, shape=(64, 64, 3), scale=[0.2, 0.8]):
    ''' unused '''
    if shape[-1] == 0:
        shape = np.random.uniform(64, 128)
        shape = (shape, shape, 3)

    crop = transforms.Compose([
        lambda x: Image.fromarray(x) if not 'PIL' in str(type(x)) else x,
        transforms.RandomResizedCrop(shape[0], scale=scale)
    ])    

    if torch.is_tensor(x):
        x = x.numpy().transpose(1,2, 0)
    
    P = []
    for _ in range(n):
        xx = transform(crop(x))
        P.append(xx)

    return torch.cat(P, dim=0)


def patch_grid(transform, shape=(64, 64, 3), stride=[0.5, 0.5]):
    stride = np.random.random() * (stride[1] - stride[0]) + stride[0]
    stride = [int(shape[0]*stride), int(shape[1]*stride), shape[2]]
    if shape[0] == 230:
        stride = [6, 6, 3]
    if shape[0] == 124:
        stride = [22, 22, 3]
    if shape[0] == 58:
        stride = [33, 33, 3]
    if shape[0] == 52:
        stride = [34, 34, 3]
    if shape[0] == 46:
        stride = [35, 35, 3]
    if shape[0] == 40:
        stride = [36, 36, 3]
    if shape[0] == 28:
        stride = [38, 38, 3]
    if shape[0] == 16:
        stride = [40, 40, 3]
    if shape[0] == 80:
        stride = [36, 36, 3]

    spatial_jitter = transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.RandomResizedCrop(shape[0], scale=(0.7, 0.9))
    ])

    def aug(x):
        if torch.is_tensor(x):
            x = x.numpy().transpose(1, 2, 0)
        elif 'PIL' in str(type(x)):
            x = np.array(x)#.transpose(2, 0, 1)
        
        winds = skimage.util.view_as_windows(x, shape, step=stride)
        winds = winds.reshape(-1, *winds.shape[-3:])

        P = [transform(spatial_jitter(w)) for w in winds]
        return torch.cat(P, dim=0)

    return aug


def get_frame_aug(frame_aug, patch_size, norm=True):
    train_transform = []

    if 'cj' in frame_aug:
        _cj = 0.1
        train_transform += [
            #transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(_cj, _cj, _cj, 0),
        ]
    if 'flip' in frame_aug:
        train_transform += [transforms.RandomHorizontalFlip()]

    if norm:
        train_transform += NORM
    train_transform = transforms.Compose(train_transform)

    print('Frame augs:', train_transform, frame_aug)

    if 'grid' in frame_aug:
        aug = patch_grid(train_transform, shape=np.array(patch_size))
    else:
        aug = train_transform

    return aug


def get_frame_transform(frame_transform_str, img_size):
    tt = []
    fts = frame_transform_str
    norm_size = torchvision.transforms.Resize((img_size, img_size))

    if 'crop' in fts:
        tt.append(torchvision.transforms.RandomResizedCrop(
            img_size, scale=(0.8, 0.95), ratio=(0.7, 1.3), interpolation=2),)
    else:
        tt.append(norm_size)

    if 'cj' in fts:
        _cj = 0.1
        # tt += [#transforms.RandomGrayscale(p=0.2),]
        tt += [transforms.ColorJitter(_cj, _cj, _cj, 0),]

    if 'flip' in fts:
        tt.append(torchvision.transforms.RandomHorizontalFlip())

    print('Frame transforms:', tt, fts)

    return tt

def get_train_transforms(opt, norm=True):
    norm_size = torchvision.transforms.Resize((opt['size'], opt['size']))

    frame_transform = get_frame_transform(opt['frame_transforms'], opt['size'])
    frame_aug = get_frame_aug(opt['frame_aug'], opt['patch_size'], norm=norm)
    if opt['frame_aug'] != '':
        frame_aug = [frame_aug]
    elif norm:
        frame_aug = NORM
    else:
        frame_aug = []
    
    transform = frame_transform + frame_aug

    train_transform = MapTransform(
            torchvision.transforms.Compose(transform)
        )

    plain_transform = [torchvision.transforms.ToPILImage(), norm_size]
    if norm:
        plain_transform = plain_transform + NORM

    plain = torchvision.transforms.Compose(plain_transform)

    def with_orig(x):
        x = train_transform(x), \
            plain(x[0]) if 'numpy' in str(type(x[0])) else plain(x[0].permute(2, 0, 1))

        return x

    return with_orig

def get_train_transforms_multiscale(opt, norm=True):
    train_transform = {}
    for size in opt['sizes']:
        norm_size = torchvision.transforms.Resize((size, size))

        frame_transform = get_frame_transform(opt['frame_transforms'], size)
        frame_aug = get_frame_aug(opt['frame_aug'], opt['patch_size'], norm=norm)
        if opt['frame_aug'] != '':
            frame_aug = [frame_aug]
        elif norm:
            frame_aug = NORM
        else:
            frame_aug = []
        
        transform = frame_transform + frame_aug

        train_transform[size] = MapTransform(
                torchvision.transforms.Compose(transform)
            )

    plain_transform = [torchvision.transforms.ToPILImage(), norm_size]
    if norm:
        plain_transform = plain_transform + NORM

    plain = torchvision.transforms.Compose(plain_transform)

    def with_orig(x):
        out = {}
        for k, v in train_transform.items():
            out[k] = v(x)
        plain_out = plain(x[0]) if 'numpy' in str(type(x[0])) else plain(x[0].permute(2, 0, 1))
        return out, plain_out

    return with_orig