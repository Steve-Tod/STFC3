import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from models.resnet import resnet18, resnet34, resnet50, BasicBlock, conv1x1
from models import resnet_nopad

class fcn_encoder(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.pretrained = opt['pretrain_res']
        self.model = opt['res_model']
        self.pool_size = opt['pool_size']
        if opt['padding'] is None:
            self.padding = 'reflect'
        else:
            self.padding = opt['padding']
        if opt['pool_type'] is None:
            pool_func = nn.AvgPool2d
        elif opt['pool_type'] == 'max':
            pool_func = nn.MaxPool2d
        elif opt['pool_type'] == 'random':
            pool_func = RandomPool
        elif opt['pool_type'] == 'random_stride':
            pool_func = RandomPoolStride
        self.proj_option = opt['proj_option']
        
        message = 'Use '
        if self.pretrained:
            message += 'pretrained '
        if self.model == 'r18':
            message += 'ResNet18.'
            self.backbone = resnet18(pretrained=self.pretrained)
            self.backbone.modify(padding=self.padding)
        elif self.model == 'r34':
            message += 'ResNet34.'
            self.backbone = resnet34(pretrained=self.pretrained)
            self.backbone.modify(padding=self.padding)
        elif self.model == 'r18_nopad':
            self.backbone = resnet_nopad.resnet18(pretrained=self.pretrained)
        elif self.model == 'r50':
            message += 'ResNet50.'
            self.backbone = resnet50(pretrained=self.pretrained)
            self.backbone.modify(padding=self.padding)
        elif self.model == 'r50_nopad':
            self.backbone = resnet_nopad.resnet50(pretrained=self.pretrained)
        else:
            raise NotImplementedError(
            'Encoder %s not recognized' % (self.model))
        if self.pool_size == 8:
            self.pool = pool_func(kernel_size=4, stride=4)
        elif self.pool_size == 7:
            self.pool = pool_func(kernel_size=8, stride=4)
        elif self.pool_size == 4:
            self.pool = pool_func(kernel_size=8, stride=8)
        else:
            self.pool = None
            
        if self.proj_option is not None:
            use_bn = self.proj_option['bn']
            self.projector = nn.Sequential()
            if self.model in ['r18', 'r18_nopad', 'r22', 'r34']:
                dim_src = 512
            else:
                dim_src = 2048
                
            for i, dim_dst in enumerate(self.proj_option['dim']):
                if i == len(self.proj_option['dim']) - 1 and self.proj_option['remove_last_bias']:
                    self.projector.add_module(name=f'conv1d_{i}_{dim_src}_{dim_dst}', module=nn.Conv2d(dim_src, dim_dst, 1, bias=False))
                else:
                    self.projector.add_module(name=f'conv1d_{i}_{dim_src}_{dim_dst}', module=nn.Conv2d(dim_src, dim_dst, 1))
                if i == len(self.proj_option['dim']) - 1:
                    break
                self.projector.add_module(name='relu_%d' % (i), module=nn.ReLU())
                if use_bn:
                    self.projector.add_module(name='bn_%d' % (i), module=nn.BatchNorm2d(dim_dst))
                dim_src = dim_dst
        
    def encode(self, x):
        x = self.backbone(x)
        return x
            
    def forward(self, x):
        x = self.backbone(x)  # (N, C, H, W)
        if self.proj_option is not None:
            x = self.projector(x)
        if self.pool is not None:
            x = self.pool(x)
        return x

def MLP_head(in_dim, dims, use_bn):
    projector = nn.Sequential()
    dim_src = in_dim
    for i, dim_dst in enumerate(dims):
        projector.add_module(name='conv1d_%d' % (i), module=nn.Conv2d(dim_src, dim_dst, 1))
        if i == len(dims) - 1:
            break
        if use_bn:
            projector.add_module(name='bn_%d' % (i), module=nn.BatchNorm2d(dim_dst))
        projector.add_module(name='relu_%d' % (i), module=nn.ReLU())
        dim_src = dim_dst
    return projector


def res18_layer4():
    downsample = nn.Sequential(
        conv1x1(256, 512, 1),
        nn.BatchNorm2d(512),
    )

    layers = []
    layers.append(BasicBlock(256, 512, 1, downsample))
    for _ in range(1, 2):
        layers.append(BasicBlock(512, 512))
    return nn.Sequential(*layers)


class Head(nn.Module):
    def __init__(self, opt):
        super().__init__()
        if opt['res_layer4']:
            self.res4 = res18_layer4()
        else:
            self.res4 = None
        if opt['mlp']:
            self.mlp = MLP_head(opt['mlp']['dim_in'], opt['mlp']['dim'], opt['mlp']['bn'])
        else:
            self.mlp = None
    def forward(self, x):
        if self.res4 is not None:
            x = self.res4(x)
        if self.mlp is not None:
            x = self.mlp(x)
        return x

class RandomPool(nn.Module):
    def __init__(self, kernel_size, stride, padding=0):
        super().__init__()
        self.unfold = nn.Unfold(kernel_size, stride=stride, padding=padding)
        self.pixel_per_pool = kernel_size ** 2
    
    def forward(self, x, T):
        # x: B, C, H, W
        B, C = x.shape[:2]
        patches = self.unfold(x)
        num_patch = patches.size(-1)
        side_len = int(np.sqrt(num_patch))
        patches = patches.view(B, C, self.pixel_per_pool, num_patch)
        
        B_batch = B // T
        select_idx = torch.randint(self.pixel_per_pool, (B_batch, 1, num_patch), device=x.device)
        select_idx = select_idx.expand(-1, T, -1).contiguous()
        # B, C, 1, P
        select_idx = select_idx.view(B, 1, 1, num_patch).expand(-1, C, -1, -1)
        # B, C, 1, P
        out = torch.gather(patches, dim=2, index=select_idx)
        
        out = out.view(B, C, side_len, side_len)
        return out

class RandomPoolStride(nn.Module):
    def __init__(self, kernel_size, stride, padding=0):
        super().__init__()
        self.unfold = nn.Unfold(kernel_size, stride=stride, padding=padding)
        self.pixel_per_pool = kernel_size ** 2
    
    def forward(self, x, T):
        # x: B, C, H, W
        B, C = x.shape[:2]
        patches = self.unfold(x)
        num_patch = patches.size(-1)
        side_len = int(np.sqrt(num_patch))
        
        patches = patches.view(B, C, self.pixel_per_pool, num_patch)
        B_batch = B // T
        select_idx = torch.randint(self.pixel_per_pool, (B_batch, 1), device=x.device)
        select_idx = select_idx.expand(-1, T).contiguous()
        # B, C, 1, P
        select_idx = select_idx.view(B, 1, 1, 1).expand(-1, C, -1, num_patch)
        # B, C, 1, P
        out = torch.gather(patches, dim=2, index=select_idx)
        
        out = out.view(B, C, side_len, side_len)
        return out