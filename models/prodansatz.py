'''
Restricted Boltzmann Machine.
'''

import numpy as np
import numbers,pdb
from scipy.special import expit

from poornn.nets import ANN
from poornn.utils import typed_randn
from poornn import SPConv, functions, Linear, Apdot

__all__=['ConvWF']

class ConvWF(ANN):
    '''
    Convolutional Ansatz for wave function.

    Attributes:
        :input_shape: tuple, (N1, N2 ...)
        :nfs: int, number features in hidden layer.
    '''
    def __init__(self, input_shape, nfs, itype='complex128'):
        assert(len(nfs)==4)
        eta=1
        nsite = np.prod(input_shape)
        self.nfs, self.itype = nfs, itype
        DIM = len(input_shape)

        POOLING_MODE = 'max-abs'

        super(ConvWF, self).__init__(itype=itype, do_shape_check=False)
        self.layers.append(functions.Reshape(input_shape, itype=itype, output_shape=(1,)+input_shape))

        self.add_layer(SPConv, weight=typed_randn(itype, (self.nfs[0], 1, nsite))*eta,
                bias=typed_randn(itype, (nfs[0],))*eta, boundary='P')
        self.add_layer(functions.Pooling, kernel_shape=(2,)*DIM, mode=POOLING_MODE)
        self.add_layer(functions.ReLU)

        self.add_layer(SPConv, weight=typed_randn(itype, (self.nfs[1], self.nfs[0], nsite))*eta,
                bias=typed_randn(itype, (nfs[1],))*eta, boundary='P')
        self.add_layer(functions.Pooling, kernel_shape=(2,)*DIM, mode=POOLING_MODE)
        self.add_layer(functions.ReLU)

        nf1_ = np.prod(self.layers[-1].output_shape)
        self.add_layer(functions.Reshape, output_shape=(nf1_,))

        self.add_layer(Apdot, weight=typed_randn(itype,(nfs[3],nfs[1]))*eta,bias=typed_randn(itype, (nfs[3],))*eta)
        self.add_layer(Linear, weight=typed_randn(itype,(1,nfs[3]))*eta,bias=typed_randn(itype, (1,))*eta)

        self.add_layer(functions.Reshape, output_shape=())
        self._shapes = None

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return '<ConvWF> in[%s] hid[%s x %s]'%(self.input_shape, self.layers[0].output_shape)

    def __call__(self, config):
        return self.get_weight(config)

    @property
    def nsite(self):
        return np.prod(self.input_shape)

    @property
    def DIM(self):
        return len(self.input_shape)

    def tovec(self,spaceconfig):  #poor designed interface.
        '''
        Get the state vector.

        \Psi(s,W)=\sum_{\{hi\}} e^{\sum_j a_j\sigma_j^z+\sum_i b_ih_i +\sum_{ij}W_{ij}h_i\sigma_j}
        '''
        configs=config=1-2*spaceconfig.ind2config(np.arange(spaceconfig.hndim))
        return np.array([self.get_weight(config_i) for config_i in configs])

    def get_weight(self,config,theta=None):
        '''
        Get the weight for specific configuration.

        Parameters:
            :config: 1darray,
            :theta: 1darray/None, table of hidden layer output: b+v.dot(W), intended to boost operation.

        Return:
            number,
        '''
        res=self.forward(config)[-1]
        return res
