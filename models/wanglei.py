'''
Restricted Boltzmann Machine.
'''

from __future__ import division
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
    def __init__(self, input_shape, num_features=[12], dtype='float64',use_conv=True,version='linear'):
        self.num_features, self.dtype = num_features, dtype
        nsite=np.prod(input_shape)
        eta=0.2
        super(WangLei, self).__init__(dtype, do_shape_check=True)

        stride = nsite
        if not use_conv:
            num_feature_hidden=num_features[0]*nsite
            self.layers.append(Linear(input_shape, dtype, weight=eta*typed_randn(self.dtype, (num_feature_hidden, nsite)),
                    bias=eta*typed_randn(self.dtype, (num_feature_hidden,))))
        else:
            self.layers.append(functions.Reshape(input_shape, dtype=dtype, output_shape=(1,)+input_shape))
            self.add_layer(SPConv, weight=eta*typed_randn(self.dtype, (self.num_features[0], 1, nsite)),
                    bias=eta*typed_randn(self.dtype, (num_features[0],)), boundary='P', strides=(stride,))
            self.add_layer(functions.Reshape, output_shape=(num_features[0], nsite//stride))
        self.add_layer(functions.Log2cosh)
        self.add_layer(functions.Sum, axis=-1)
        self.add_layer(functions.Exp)
        if version=='const-linear': 
            self.add_layer(Linear, weight=np.array([[-1,-1,1,1]],dtype=dtype, order='F'),
                    bias=np.zeros((1,),dtype=dtype),var_mask=(0,0))
        elif version=='linear':
            for nfi, nfo in zip(num_features, num_features[1:]+[1]):
                self.add_layer(Linear, weight=eta*typed_randn(self.dtype, (nfo, nfi)),
                        bias=eta*typed_randn(self.dtype, (nfo,)),var_mask=(1,1))
        elif version=='rbm':
            pass
        else:
            raise ValueError('version %s not exist'%version)
        self.add_layer(functions.Reshape, output_shape=())
