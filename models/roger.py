'''
Restricted Boltzmann Machine.
'''

import numpy as np
import pdb

from poornn.utils import typed_randn
from poornn import SPConv, Linear, functions
from qstate.classifier import PSNN
from qstate import StateNN

__all__=['Roger']

class Roger(StateNN):
    '''
    Restricted Boltzmann Machine class.

    Attributes:
        :input_shape: tuple, (1, N1, N2 ...)
        :num_feature_hidden: int, number features in hidden layer.
        :use_msr: use marshall sign rule.
    '''
    def __init__(self, input_shape, num_feature_hidden, with_linear=False, use_msr=False, theta_period=2):
        self.num_feature_hidden = num_feature_hidden
        self.use_msr = use_msr
        dtype = 'float64'
        self.with_linear = with_linear
        nsite=np.prod(input_shape)
        eta=0.1
        super(Roger, self).__init__(dtype, do_shape_check=False)

        self.layers.append(functions.Reshape(input_shape, dtype=dtype, output_shape=(1,)+input_shape))
        self.add_layer(SPConv, weight=eta*typed_randn(dtype, (self.num_feature_hidden, 1, nsite)),
                bias=eta*typed_randn(dtype, (num_feature_hidden,)), boundary='P')
        self.add_layer(functions.Log2cosh)
        self.add_layer(functions.Reshape, output_shape=(num_feature_hidden, nsite) if with_linear else (num_feature_hidden*nsite,))
        self.add_layer(functions.Sum, axis=-1)
        self.add_layer(functions.Exp)
        if with_linear:
            self.add_layer(Linear, weight=eta*typed_randn(dtype, (128, self.num_feature_hidden)),
                    bias=0*typed_randn(dtype, (128,)))
            self.add_layer(functions.ReLU)
            self.add_layer(Linear, weight=eta*typed_randn(dtype, (64, 128)),
                    bias=0*typed_randn(dtype, (64,)))
            self.add_layer(functions.ReLU)
            self.add_layer(Linear, weight=eta*typed_randn(dtype, (32, 64)),
                    bias=0*typed_randn(dtype, (32,)))
            self.add_layer(functions.ReLU)
            self.add_layer(Linear, weight=eta*typed_randn(dtype, (1, 32)),
                    bias=0*typed_randn(dtype, (1,)))
            self.add_layer(functions.Reshape, output_shape=())

        if use_msr and theta_period!=2:
            raise ValueError()
        self.thnn = PSNN(input_shape, period=theta_period, batch_wise=False, output_mode='theta', use_msr=use_msr)

    def get_sign(self, config, return_thys=False ,**kwargs):
        '''Get sign using sign network.'''
        thys = self.thnn.forward(config)
        if return_thys:
            return np.exp(1j*thys[-1]), thys
        else:
            return np.exp(1j*thys[-1])

    def get_variables(self):
        return np.concatenate([super(Roger,self).get_variables(), self.thnn.get_variables()])

    def set_variables(self, v):
        nv1=super(Roger, self).num_variables
        super(Roger, self).set_variables(v[:nv1])
        self.thnn.set_variables(v[nv1:])

    @property
    def num_variables(self):
        return super(Roger, self).num_variables+self.thnn.num_variables
