'''
Restricted Boltzmann Machine.
'''

import numpy as np
import pdb

from qstate import StateNN
from poornn.utils import typed_randn
from poornn import SPConv, Linear, functions, ParallelNN

__all__=['WangLei']

class WangLei3(StateNN):
    '''
    Restricted Boltzmann Machine class.

    Attributes:
        :input_shape: tuple, (1, N1, N2 ...)
        :num_feature_hidden: int, number features in hidden layer.
    '''
    def __init__(self, input_shape, num_features=[12], dtype='float64',version='linear'):
        self.num_features, self.dtype = num_features, dtype
        nsite=np.prod(input_shape)
        eta=0.2
        super(WangLei3, self).__init__(dtype, do_shape_check=False)

        D = len(input_shape)
        ishape = (1,)+input_shape
        stride = 1
        NF = 3
        self.layers.append(functions.Reshape(input_shape, dtype=dtype, output_shape=ishape))
        plnn = ParallelNN(input_shape = ishape, output_shape=(1,NF,)+tuple([si/stride for si in ishape[1:]]), dtype=dtype, axis=1)
        plnn.add_layer(functions.Reshape)
        plnn.add_layer(functions.ConvProd, powers=[1,1], boundary='P', strides=(stride,)*D)
        plnn.add_layer(functions.ConvProd, powers=[1,0,1], boundary='P', strides=(stride,)*D)
        self.layers.append(plnn)
        #self.add_layer(functions.ConvProd, kernel_shape=(2,)*D, boundary='P', strides=(stride,)*D)
        if version=='linear':
            self.add_layer(functions.Reshape, output_shape=(np.prod(self.layers[-1].output_shape),))
            self.add_layer(Linear, weight=eta*typed_randn(self.dtype, (num_features[0], self.layers[-1].output_shape[-1])),
                    bias=eta*typed_randn(self.dtype, (num_features[0],)),var_mask=(1,1))
        elif version=='conv':
            stride= 1
            imgsize = self.layers[-1].output_shape[-1]
            self.add_layer(SPConv, weight=eta*typed_randn(self.dtype, (self.num_features[0], NF, imgsize)),
                    bias=eta*typed_randn(self.dtype, (num_features[0],)), boundary='P', strides=(stride,)*D)
            self.add_layer(functions.Reshape, output_shape=(num_features[0], imgsize/stride**D))

            self.add_layer(functions.Power,order=3)
            self.add_layer(functions.Sum, axis=-1)
        if version=='const-linear': 
            self.add_layer(Linear, weight=np.array([[-1,-1,1,1]],dtype=dtype, order='F'),
                    bias=np.zeros((1,),dtype=dtype),var_mask=(0,0))
        elif version=='linear' or version=='conv':
            for nfi, nfo in zip(num_features, num_features[1:]+[1]):
                self.add_layer(Linear, weight=eta*typed_randn(self.dtype, (nfo, nfi)),
                        bias=eta*typed_randn(self.dtype, (nfo,)),var_mask=(1,1))
        elif version=='rbm':
            pass
        else:
            raise ValueError('version %s not exist'%version)
        self.add_layer(functions.Reshape, output_shape=())
        #self.add_layer(functions.Exp)
