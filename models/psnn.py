from __future__ import division
import numpy as np
import pdb

from poornn.nets import ANN
from poornn.utils import typed_randn
from poornn import SPConv, Linear, functions

from qstate.classifier import ThetaNN

class PSNN(ThetaNN):
    '''
    Periodic NN to determine sign.

    Attributes:
        :input_shape: tuple, (N1, N2 ...)
        :num_feature_hidden: int, number features in hidden layer.
        :output_mode: 'sign'/'loss'/'theta'.
    '''
    def __init__(self, input_shape, period, kernel='cos',nf=4, batch_wise=False, output_mode='theta', use_msr=False):
        self.period = period
        self.batch_wise=batch_wise
        if batch_wise:
            num_batch=input_shape[0]
            site_shape = input_shape[1:]
        else:
            num_batch = 1
            site_shape = input_shape
        nsite=np.prod(site_shape)
        eta=0.1
        super(PSNN, self).__init__('float64' if output_mode=='theta' else 'complex128')

        dtype = 'float64'
        self.layers.append(functions.Reshape(input_shape, itype='float64', output_shape=(num_batch,1)+site_shape))
        if use_msr:
            weight=np.array([[[np.pi/2,0]]])
            bias=np.array([np.pi/2])
            var_mask=(0,0)
        else:
            weight=eta*typed_randn('float64', (nf, 1, nsite))
            bias=eta*typed_randn('float64', (nf, ))
            var_mask=(1,1)
        self.add_layer(SPConv, weight=weight, bias=bias, strides=(period,), boundary='P', var_mask=var_mask)
        self.add_layer(functions.Reshape, output_shape=(num_batch,nf,nsite//period))
        #self.add_layer(functions.Cos)
        self.add_layer(functions.Sum, axis=-1)
        self.add_layer(functions.ReLU)
        self.add_layer(Linear, weight=eta*typed_randn(dtype, (1, nf)),
                bias=0*typed_randn(dtype, (1,)))

        if output_mode != 'theta':
            if kernel == 'exp':
                self.add_layer(functions.TypeCast, otype='complex128')
                self.add_layer(functions.Mul,alpha=1j)
                self.add_layer(functions.Exp)
                self.add_layer(functions.TypeCast, otype='float64')
            elif kernel == 'cos':
                self.add_layer(functions.Cos)
        if output_mode == 'loss':
            self.add_layer(functions.SquareLoss)
            self.add_layer(functions.Mean,axis=0)

        if not batch_wise:
            self.add_layer(functions.Reshape, output_shape=())
