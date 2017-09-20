'''
Restricted Boltzmann Machine.
'''

from __future__ import division
import numpy as np
import pdb

from qstate import StateNN
from poornn.utils import typed_randn
from poornn import SPConv, Linear, functions, ParallelNN

__all__=['WangLei5']

class WangLei5(StateNN):
    '''
    Restricted Boltzmann Machine class.

    Attributes:
        :input_shape: tuple, (N1, N2 ...)
        :NF: int, number of features in product layer.
        :K: int, filter width in product layer.
        :num_features: int, number features in hidden layer.
        :eta0, eta1: float, variance of initial variables in product/linear layers.
        :dtype0, dtype1: str, data type of variables in product/linear layers.
        :itype: str, input data dtype.
        :version: ['linear'|'conv'|'const-linear'],
        :stride: int, stride step in convolusion layers.
    '''
    def __init__(self, input_shape, K=2, num_features=[4,4,4], eta0=0.2, eta1=0.2, NP=1, NC=1,
            itype='complex128',version='linear', dtype0='complex128', dtype1='complex128', stride=None, usesum=False):
        self.num_features, self.itype = num_features, itype
        if stride is None:
            if any([n%4!=0 for n in input_shape]):
                stride=2
            else:
                stride=1
        self.stride = stride
        nsite=np.prod(input_shape)
        super(WangLei5, self).__init__(itype, do_shape_check=False)

        D = len(input_shape)
        ishape = (1,)+input_shape
        self.layers.append(functions.Reshape(input_shape, itype=itype, output_shape=ishape))
        imgsize = self.layers[-1].output_shape[-D:]

        # product layers.
        dtype = dtype0
        eta=eta0
        self.add_layer(functions.Log)
        for nfi, nfo in zip([1]+num_features[:NP-1], num_features[:NP]):
            self.add_layer(SPConv, weight=eta*typed_randn(dtype, (nfo, nfi)+(K,)*D),
                    bias=eta*typed_randn(dtype, (nfo,)), boundary='P', strides=(stride,)*D)
            imgsize = self.layers[-1].output_shape[-D:]
        self.add_layer(functions.Exp)

        # convolution layers.
        eta=eta1
        dtype = dtype1
        stride = 1
        for nfi, nfo in zip(num_features[NP-1:NP+NC-1], num_features[NP:NP+NC]):
            self.add_layer(SPConv, weight=eta*typed_randn(dtype, (nfo, nfi)+imgsize),
                    bias=eta*typed_randn(dtype, (nfo,)), boundary='P', strides=(stride,)*D)
            imgsize = self.layers[-1].output_shape[-D:]
        self.add_layer(functions.Reshape, output_shape=(nfo,np.prod(imgsize)//stride**D))

        # non-linear function
        self.add_layer(functions.Power,order=3)
        self.add_layer(functions.Mean, axis=-1)

        # linear layers.
        if usesum:
            self.add_layer(functions.Sum, axis=-1)
        else:
            for i,(nfi, nfo) in enumerate(zip(num_features[NP+NC-1:], num_features[NP+NC:]+[1])):
                if i!=0:
                    self.add_layer(functions.Log2cosh)
                self.add_layer(Linear, weight=eta*typed_randn(dtype, (nfo, nfi)),
                        bias=eta*typed_randn(dtype, (nfo,)),var_mask=(1,1))
            self.add_layer(functions.Reshape, output_shape=())

