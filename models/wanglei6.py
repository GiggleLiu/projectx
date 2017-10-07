'''
Restricted Boltzmann Machine.
'''

from __future__ import division
import numpy as np
import pdb

from qstate import StateNN
from poornn.utils import typed_uniform
from poornn import SPConv, Linear, functions, ParallelNN, layers

__all__=['WangLei6']

class WangLei6(StateNN):
    '''
    Restricted Boltzmann Machine class.

    Attributes:
        :input_shape: tuple, (N1, N2 ...)
        :K: int, filter width in product layer.
        :num_features: int, number features in hidden layer.
        :eta0, eta1: float, variance of initial variables in product/linear layers.
        :dtype0, dtype1: str, data type of variables in product/linear layers.
        :itype: str, input data dtype.
        :stride: int, stride step in convolusion layers.
    '''
    def __init__(self, input_shape, powerlist=None, num_features=[4,4,4], eta0=0.2, eta1=0.2, NP=1, NC=1, K=None,\
            itype='complex128',dtype0='complex128', dtype1='complex128', momentum=0.,
                    stride=None, usesum=False, nonlinear='x^3',poly_order=10, do_BN=True):
        if K is None:
            K=np.prod(input_shape)
        self.num_features, self.itype = num_features, itype
        self.stride = stride
        nsite=np.prod(input_shape)
        D = len(input_shape)
        ishape = (1,)+input_shape

        super(WangLei6, self).__init__(layers = [functions.Reshape(input_shape, itype=itype, output_shape=ishape)])

        # preprocessing
        if powerlist is not None:
            nfo = len(powerlist)
            plnn = ParallelNN(axis=1)
            for power in powerlist:
                plnn.layers.append(functions.ConvProd(ishape, itype, powers=power, boundary='P', strides=(stride,)*D))
            self.layers.append(plnn)
        else:
            nfo = 1

        # product layers.
        dtype = dtype0
        eta=eta0
        if NP!=0: self.add_layer(functions.Log)
        for nfi, nfo in zip([nfo]+num_features[:NP-1], num_features[:NP]):
            self.add_layer(SPConv, weight=eta*typed_uniform(dtype, (nfo, nfi)+(K,)*D),
                    bias=eta*typed_uniform(dtype, (nfo,)), boundary='P', strides=(stride,)*D)
            input_shape = self.layers[-1].output_shape[-D:]
        if NP!=0: self.add_layer(functions.Exp)

        # convolution layers.
        eta=eta1
        dtype = dtype1
        stride = 1
        for nfi, nfo in zip([nfo]+num_features[NP:NP+NC-1], num_features[NP:NP+NC]):
            self.add_layer(SPConv, weight=eta*typed_uniform(dtype, (nfo, nfi)+input_shape),
                    bias=eta*typed_uniform(dtype, (nfo,)), boundary='P', strides=(stride,)*D)
            input_shape = self.layers[-1].output_shape[-D:]
        self.add_layer(functions.Reshape, output_shape=(nfo,np.prod(input_shape)//stride**D))

        # non-linear function
        if nonlinear=='x^3':
            self.add_layer(functions.Power,order=3)
        elif nonlinear=='x^5':
            self.add_layer(functions.Power,order=5)
        elif nonlinear=='relu':
            self.add_layer(functions.ReLU)
        elif nonlinear=='sinh':
            self.add_layer(functions.Sinh)
        elif nonlinear in layers.Poly.kernel_dict:
            self.add_layer(layers.Poly, params=eta1*typed_uniform('complex128', (poly_order,)), kernel=nonlinear)
        else:
            raise Exception
        self.add_layer(functions.Filter, axes=(-1,), momentum=momentum)

        # linear layers.
        if usesum:
            self.add_layer(functions.Mean, axis=-1)
        else:
            for i,(nfi, nfo) in enumerate(zip(num_features[NP+NC-1:], num_features[NP+NC:]+[1])):
                if do_BN:
                    self.add_layer(functions.BatchNorm, axis=None, label='BN-%s'%i)
                    self.add_layer(layers.Poly, params=np.array([0j,1.]), kernel='polynomial')
                self.add_layer(functions.Sinh)
                self.add_layer(Linear, weight=eta*typed_uniform(dtype, (nfo, nfi)),
                        bias=eta*typed_uniform(dtype, (nfo,)),var_mask=(1,1))
            self.add_layer(functions.Reshape, output_shape=())

