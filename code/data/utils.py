import torch


def generate_neighbour_label(h, w, dist_type, rad):
    hs = torch.arange(end=h)
    ws = torch.arange(end=w)
    h_grid, w_grid = torch.meshgrid(hs, ws)
    grid = torch.stack((h_grid, w_grid), dim=-1).view(-1, 1, 2).float()
    grid_ = grid.transpose(0, 1)
    dist = torch.norm(grid - grid_, p=dist_type, dim=-1)
    label = dist <=  rad
    return label

def generate_smooth_label(size, smooth_type, max_prob=0.9, temp=0.1, rad=0.3, dist_type=1):
    if smooth_type == 'onehot':
        hs = torch.arange(end=size)
        ws = torch.arange(end=size)
        h_grid, w_grid = torch.meshgrid(hs, ws)
        grid = torch.stack((h_grid, w_grid), dim=-1).view(-1, 1, 2).float()
        grid_ = grid.transpose(0, 1)
        dist = torch.norm(grid - grid_, p=dist_type, dim=-1)
        dist /= size
        label = torch.zeros_like(dist)
        label[dist <= rad] = 1
        other_prob = (1 - max_prob) / (label.sum(-1) - 1)
        label *= other_prob.unsqueeze(-1)
        label[torch.arange(size * size), torch.arange(size * size)] = max_prob
    elif smooth_type == 'softmax':
        hs = torch.arange(end=size)
        ws = torch.arange(end=size)
        h_grid, w_grid = torch.meshgrid(hs, ws)
        grid = torch.stack((h_grid, w_grid), dim=-1).view(-1, 1, 2).float()
        grid_ = grid.transpose(0, 1)
        dist = torch.norm(grid - grid_, p=dist_type, dim=-1)
        dist /= size
        dist[dist > rad] = 1e10
        label = torch.softmax(- dist / temp, dim=-1)
    return label