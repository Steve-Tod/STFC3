import logging
logger = logging.getLogger('base')

def define_net(opt_net, rank):
    which_model = opt_net['model_type']

    if which_model == 'STFC3Net':
        from .STFC3Net import STFC3Net as m
    else:
        raise NotImplementedError(
            'Model [{:s}] not recognized'.format(which_model))
        
    net = m(opt_net)
    if rank == 0:
        logger.info('Model [%s] created' % which_model)

    return net
