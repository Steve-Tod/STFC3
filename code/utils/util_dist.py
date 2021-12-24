import os
import logging
import torch.distributed as dist

logger = logging.getLogger('base')

def dist_init(rank, backend='nccl'):
    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
    }
    logger.info(f"[{os.getpid()}] Initializing process group with: {env_dict}")
    dist.init_process_group(backend="nccl", init_method='env://')
    assert rank == dist.get_rank()
    logger.info(
        f"[{os.getpid()}]: world_size = {dist.get_world_size()}, "
        + f"rank = {dist.get_rank()}, backend={dist.get_backend()}"
    )
    return dist.get_world_size()
    
def dist_barrier(opt):
    if opt['distributed']:
        dist.barrier()
        
def dist_destroy(opt):
    if opt['distributed']:
        dist.destroy_process_group()
