'''
Restricted Boltzmann Machine.
'''

import numpy as np
import numbers,pdb
from scipy.special import expit

from poornn.utils import typed_randn
from poornn import SPConv, functions
from qstate import StateNN

__all__=['RBM']

class RBM(StateNN):
    '''
    Restricted Boltzmann Machine class.

    Attributes:
        :input_shape: tuple, (1, N1, N2 ...)
        :num_feature_hidden: int, number features in hidden layer.
    '''
    def __init__(self, input_shape, num_feature_hidden, itype='complex128', sign_func=None):
        self.num_feature_hidden, self.itype = num_feature_hidden, itype
        nsite=np.prod(input_shape)
        eta=0.1
        super(RBM, self).__init__(itype, do_shape_check=False)

        self.layers.append(functions.Reshape(input_shape, itype=itype, output_shape=(1,)+input_shape))
        self.add_layer(SPConv, weight=eta*typed_randn(self.itype, (self.num_feature_hidden, 1, nsite)),
                bias=eta*typed_randn(self.itype, (num_feature_hidden,)), boundary='P')
        self.add_layer(functions.Log2cosh)
        self.add_layer(functions.Reshape, output_shape=(self.num_feature_hidden*nsite,))
        self.add_layer(functions.Sum, axis=0)
        self.add_layer(functions.Exp)

        self._get_sign = sign_func

    def get_sign(self, config):
        if self._get_sign is None:
            return 1
        else:
            return self._get_sign(config)
