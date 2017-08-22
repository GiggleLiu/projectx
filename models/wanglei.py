'''
Restricted Boltzmann Machine.
'''

import numpy as np
import pdb

from qstate import StateNN
from poornn.utils import typed_randn
from poornn import SPConv, Linear, functions

__all__=['WangLei']

class WangLei(StateNN):
    '''
    Restricted Boltzmann Machine class.

    Attributes:
        :input_shape: tuple, (1, N1, N2 ...)
        :num_feature_hidden: int, number features in hidden layer.
    '''
    def __init__(self, input_shape, num_feature_hidden, dtype='float64',linear_version=True):
        self.num_feature_hidden, self.dtype = num_feature_hidden, dtype
        nsite=np.prod(input_shape)
        eta=0.05
        super(WangLei, self).__init__(dtype, do_shape_check=True)

        stride = nsite
        if linear_version:
            num_feature_hidden=num_feature_hidden*nsite
            #self.layers.append(functions.Reshape(input_shape, dtype=dtype, output_shape=(1,)+input_shape))
            self.layers.append(Linear(input_shape, dtype, weight=eta*typed_randn(self.dtype, (num_feature_hidden, nsite)),
                    bias=eta*typed_randn(self.dtype, (num_feature_hidden,))))
        else:
            self.layers.append(functions.Reshape(input_shape, dtype=dtype, output_shape=(1,)+input_shape))
            self.add_layer(SPConv, weight=eta*typed_randn(self.dtype, (self.num_feature_hidden, 1, nsite)),
                    bias=eta*typed_randn(self.dtype, (num_feature_hidden,)), boundary='P', strides=(stride,))
            self.add_layer(functions.Reshape, output_shape=(num_feature_hidden, nsite/stride))
        self.add_layer(functions.Log2cosh)
        self.add_layer(functions.Sum, axis=-1)
        self.add_layer(functions.Exp)
        #self.add_layer(Linear, weight=eta*typed_randn(self.dtype, (1, self.num_feature_hidden)),
                #bias=eta*typed_randn(self.dtype, (1,)))
        if not linear_version: 
            self.add_layer(Linear, weight=np.array([[-1,-1,1,1]],dtype=dtype, order='F'),
                    bias=np.zeros((1,),dtype=dtype),var_mask=(False,True))
            self.add_layer(functions.Reshape, output_shape=())
