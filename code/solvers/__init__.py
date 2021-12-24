import logging

logger = logging.getLogger('base')

def create_solver(opt):
    solver_type = opt['type']
    if solver_type == 'STFC3Solver':
        from .STFC3Solver import STFC3Solver as S
    else:
        raise NotImplementedError(
            'Solver [%s] not recognized' % solver_type)
    solver = S(opt)
    if opt['rank'] == 0:
        logger.info('Solver [%s] created' % solver_type)
    return solver
