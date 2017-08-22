'''
Restricted Boltzmann Machine.
'''

import numpy as np
import pdb

from poornn.utils import typed_randn
from poornn import SPConv, functions, Linear
from qstate import StateNN

__all__=['ConvWF']

class ConvWF(StateNN):
    '''
    Convolutional Ansatz for wave function.

    Attributes:
        :input_shape: tuple, (N1, N2 ...)
        :nfs: int, number features in hidden layer.
    '''
    def __init__(self, input_shape, nfs, dtype='complex128'):
        assert(len(nfs)==4)
        eta=0.1
        nsite = np.prod(input_shape)
        self.nfs, self.dtype = nfs, dtype
        DIM = len(input_shape)

        POOLING_MODE = 'mean'

        super(ConvWF, self).__init__(dtype=dtype, do_shape_check=False)
        self.layers.append(functions.Reshape(input_shape, dtype=dtype, output_shape=(1,)+input_shape))

        self.add_layer(SPConv, weight=typed_randn(dtype, (self.nfs[0], 1, nsite))*eta,
                bias=typed_randn(dtype, (nfs[0],))*eta, boundary='P')
        self.add_layer(functions.Pooling, kernel_shape=(2,)*DIM, mode=POOLING_MODE)
        self.add_layer(functions.ReLU)

        self.add_layer(SPConv, weight=typed_randn(dtype, (self.nfs[1], self.nfs[0], nsite))*eta,
                bias=typed_randn(dtype, (nfs[1],))*eta, boundary='P')
        self.add_layer(functions.Pooling, kernel_shape=(2,)*DIM, mode=POOLING_MODE)
        self.add_layer(functions.ReLU)

        nf1_ = np.prod(self.layers[-1].output_shape)
        self.add_layer(functions.Reshape, output_shape=(nf1_,))

        self.add_layer(Linear, weight=typed_randn(dtype, (nfs[2], nfs[1]*np.prod([ni/4 for ni in input_shape])))*eta,
                bias=typed_randn(dtype, (nfs[2],))*eta)
        self.add_layer(functions.ReLU)

        self.add_layer(Linear, weight=typed_randn(dtype,(nfs[3],nfs[2]))*eta,bias=typed_randn(dtype, (nfs[3],))*eta)
        self.add_layer(functions.ReLU)

        self.add_layer(Linear, weight=typed_randn(dtype,(1,nfs[3]))*eta,bias=typed_randn(dtype, (1,))*eta)
        self.add_layer(functions.Reshape, output_shape=())
        #self.add_layer(functions.Exp)

        self._shapes = None

    @property
    def DIM(self):
        return len(self.input_shape)
