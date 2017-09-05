'''
Restricted Boltzmann Machine.
'''

from __future__ import division
import numpy as np
import pdb

from qstate import StateNN
from poornn.utils import typed_randn
from poornn import SPConv, Linear, functions, ParallelNN

__all__=['WangLei4']

class WangLei4(StateNN):
    '''
    Restricted Boltzmann Machine class.

    Attributes:
        :input_shape: tuple, (1, N1, N2 ...)
        :num_feature_hidden: int, number features in hidden layer.
    '''
    def __init__(self, input_shape, NF=4, K=2, num_features=[12], itype='float64',version='linear', stride=None):
        self.num_features, self.itype = num_features, itype
        if stride is None:
            if any([n%4!=0 for n in input_shape]):
                stride=2
            else:
                stride=1
        self.stride = stride
        nsite=np.prod(input_shape)
        eta=0.2
        super(WangLei4, self).__init__(itype, do_shape_check=False)

        D = len(input_shape)
        ishape = (1,)+input_shape
        self.layers.append(functions.Reshape(input_shape, itype=itype, output_shape=ishape))

        dtype = 'float64'
        self.add_layer(functions.Log)
        #self.add_layer(SPConv, weight=eta*typed_randn(self.itype, (NF,1)+(K,)*D),
        #        bias=eta*typed_randn(self.itype, (NF,)), boundary='P', strides=(stride,)*D)
        self.add_layer(SPConv, weight=eta*typed_randn(dtype, (NF,1)+(K,)*D),
                bias=eta*typed_randn(dtype, (NF,)), boundary='P', strides=(stride,)*D)
        self.add_layer(functions.Exp)

        if version=='linear':
            self.add_layer(functions.Reshape, output_shape=(np.prod(self.layers[-1].output_shape),))
            self.add_layer(Linear, weight=eta*typed_randn(self.itype, (num_features[0], self.layers[-1].output_shape[-1])),
                    bias=eta*typed_randn(self.itype, (num_features[0],)),var_mask=(1,1))
        elif version=='conv':
            stride= 1
            imgsize = self.layers[-1].output_shape[-D:]
            self.add_layer(SPConv, weight=eta*typed_randn(self.itype, (self.num_features[0], NF)+imgsize),
                    bias=eta*typed_randn(self.itype, (num_features[0],)), boundary='P', strides=(stride,)*D)
            self.add_layer(functions.Reshape, output_shape=(num_features[0], np.prod(imgsize)//stride**D))

            self.add_layer(functions.Power,order=3)
            #self.add_layer(functions.Log2cosh)
            self.add_layer(functions.Mean, axis=-1)
        if version=='const-linear': 
            self.add_layer(Linear, weight=np.array([[-1,-1,1,1]],dtype=itype, order='F'),
                    bias=np.zeros((1,),dtype=itype),var_mask=(0,0))
        elif version=='linear' or version=='conv':
            for i,(nfi, nfo) in enumerate(zip(num_features, num_features[1:]+[1])):
                if i!=0:
                    self.add_layer(functions.ReLU)
                self.add_layer(Linear, weight=eta*typed_randn(self.itype, (nfo, nfi)),
                        bias=eta*typed_randn(self.itype, (nfo,)),var_mask=(1,1))
        elif version=='rbm':
            pass
        else:
            raise ValueError('version %s not exist'%version)
        self.add_layer(functions.Reshape, output_shape=())

