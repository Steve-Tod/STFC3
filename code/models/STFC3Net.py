import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as cp
import numpy as np
from .encoders import fcn_encoder

class STFC3Net(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.encoder = fcn_encoder(opt) 
        if opt['dropout'] is not None:
            self.dropout = opt['dropout']
        else:
            self.dropout = 0
        self.softmax = nn.Softmax(-1)
        self.temp = opt['temperature']
        self.checkpoint = opt['checkpoint']
    
    def align_feat(self, feat, affine_mat):
        BT, feats_dim, feats_h, feats_w = feat.shape
        affine_grid = nn.functional.affine_grid(affine_mat, (BT, feats_dim, feats_h, feats_w), True)
        aligned_feat = nn.functional.grid_sample(feat, 
                                                affine_grid, 
                                                mode='bilinear', 
                                                align_corners=True)
        return aligned_feat
            
    def forward(self, x, affine_mat_in):
        B, T, C, H, W = x.shape
        T = int(T  / 2)
        if len(affine_mat_in.shape) == 3:
            # B,1,2,3
            affine_mat = affine_mat_in.unsqueeze(1)
            # B,T,2,3
            affine_mat = affine_mat.expand(-1, T, -1,-1).contiguous()
        elif len(affine_mat_in.shape) == 4:
            affine_mat = affine_mat_in
        affine_mat = affine_mat.view(B*T, 2, 3)

        frames = x.view(B*2*T, C, H, W)
        # feat_map: B*(2*T-1), C1, h1, w1  e.g., B*T, c1, 8, 8
        if self.checkpoint:
            frames += torch.zeros(1, requires_grad=True).float().to(frames_forward.device)
            feat = cp(self.encoder, frames)
        else:
            feat = self.encoder(frames)

        feat_dim, feat_h, feat_w = feat.shape[1:]
        
        feat = nn.functional.normalize(feat, p=2, dim=1)
        feat = feat.view(B, 2*T, feat_dim, feat_h, feat_w)
        
        feat_forward = feat[:, :T].contiguous().view(B*T, feat_dim, feat_h, feat_w)
        feat_backward = feat[:, T:].contiguous().view(B*T, feat_dim, feat_h, feat_w)
        feat_forward_aligned = self.align_feat(feat_forward, affine_mat)
        feat_forward_aligned = nn.functional.normalize(feat_forward_aligned, p=2, dim=1)
        
        #B,T,h1*w1,C
        feat_forward_aligned = feat_forward_aligned.permute(0, 2, 3, 1).contiguous().view(B, T, feat_h * feat_w, feat_dim)
        feat_forward = feat_forward.permute(0, 2, 3, 1).contiguous().view(B, T, feat_h * feat_w, feat_dim)
        feat_backward = feat_backward.permute(0, 2, 3, 1).contiguous().view(B, T, feat_h * feat_w, feat_dim)
        
        # B,2T,h1*w1,C
        feat = torch.cat((feat_forward, feat_backward), dim=1)
        
        # compute transition matrix aligned
        # B,2T-1,P,P
        AA = torch.matmul(feat[:, :-1], feat[:, 1:].transpose(-1, -2))
        # B,T-1,P,P
        A_aligned = torch.matmul(feat_forward_aligned[:, :-1], feat_forward[:, 1:].transpose(-1, -2))

        if self.dropout > 0:
            AA[torch.rand_like(AA) < self.dropout] = -1e20
            A_aligned[torch.rand_like(A_aligned) < self.dropout] = -1e20
        AA = self.softmax(AA / self.temp)
        A_aligned = self.softmax(A_aligned / self.temp)
        At_list = []

        # affinity matrix T-1 -> T-1'
        transition_mat = AA[:, T-1]
        for t_sub in range(0, T-1):
            src = T - 2 - t_sub
            dst = T + t_sub
            tmp_mat = transition_mat @ AA[:, dst]
            # Compute transition matrix, the first affinity matrix is from aligned frames
            if t_sub > 0:
                # only use cycles with length > 1
                # longer cycle comes first
                At_list.insert(0, A_aligned[:, src] @ tmp_mat)
            transition_mat = AA[:, src] @ tmp_mat
        return At_list
