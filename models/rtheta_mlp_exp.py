'''
Restricted Boltzmann Machine.
'''

import numpy as np
import pdb

from poornn.utils import typed_randn
from poornn import SPConv, Linear, functions
#from qstate.classifier import PSNN
from psnn import PSNN
from qstate import StateNN

__all__=['RTheta_MLP_EXP']

class RTheta_MLP_EXP(StateNN):
    '''
    Restricted Boltzmann Machine class.

    Attributes:
        :input_shape: tuple, (1, N1, N2 ...)
        :num_feature_hidden: int, number features in hidden layer.
        :use_msr: use marshall sign rule.
    '''
    def __init__(self, input_shape, num_feature_hidden, mlp_shape, use_msr=False, theta_period=2):
        self.num_feature_hidden = num_feature_hidden
        self.use_msr = use_msr
        dtype = 'float64'
        nsite=np.prod(input_shape)
        eta=0.1
        super(RTheta_MLP_EXP, self).__init__(dtype, do_shape_check=False)

        self.layers.append(functions.Reshape(input_shape, dtype=dtype, output_shape=(1,)+input_shape))
        self.add_layer(SPConv, weight=eta*typed_randn(dtype, (self.num_feature_hidden, 1, nsite)),
                bias=eta*typed_randn(dtype, (num_feature_hidden,)), boundary='P')
        self.add_layer(functions.Log2cosh)
        self.add_layer(functions.Reshape, output_shape=(num_feature_hidden, nsite))
        self.add_layer(functions.Sum, axis=-1)
        self.add_layer(functions.Exp)
        self.add_layer(Linear, weight=eta*typed_randn(dtype, (mlp_shape[0], self.num_feature_hidden)),
                bias=0*typed_randn(dtype, (mlp_shape[0],)))
        self.add_layer(functions.ReLU)
        for i in range(len(mlp_shape)-1):
            self.add_layer(Linear, weight=eta*typed_randn(dtype, (mlp_shape[i+1], mlp_shape[i])),
                    bias=0.1*typed_randn(dtype, (mlp_shape[i+1],)))
            self.add_layer(functions.ReLU)
        self.add_layer(Linear, weight=eta*typed_randn(dtype, (1, mlp_shape[-1])),
                bias=0*typed_randn(dtype, (1,)))
        # self.add_layer(functions.ReLU)
        self.add_layer(functions.Exp)
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
        return np.concatenate([super(RTheta_MLP_EXP,self).get_variables(), self.thnn.get_variables()])

    def set_variables(self, v):
        nv1=super(RTheta_MLP_EXP, self).num_variables
        super(RTheta_MLP_EXP, self).set_variables(v[:nv1])
        self.thnn.set_variables(v[nv1:])

    @property
    def num_variables(self):
        return super(RTheta_MLP_EXP, self).num_variables+self.thnn.num_variables
