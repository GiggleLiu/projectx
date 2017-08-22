import numpy as np

class Sx(LinOp):
    '''sigma_x'''
    def __init__(self,i):
        self.i=i

    def loc(self, config, state, runtime, **kwargs):
        nc=np.copy(config)
        nc[self.i]*=-1
        return state.get_weight(nc)/runtime['ys'][-1]

class AFMSF(LinOp):
    '''
    AFM Structural factor.

    S_{AF}(x) = 1/N \sum_{i,j} (-1)^{i+j} s_i s_j
    '''
    def loc(self, config, state, runtime, **kwargs):
        '''
        <x|S_{AF}|\Psi>/<x|\Psi> = S_{AF}(x)
        '''
        nsite=config.shape[-1]
        #Q: why /nsite, not /nsite**2 ?
        return ((-1)**np.arange(nsite)*config)**2/nsite
