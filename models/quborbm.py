'''
Restricted Boltzmann Machine.
'''

import numpy as np
import numbers,pdb
from scipy.special import expit

from poornn.nets import ANN
from poornn.utils import typed_randn
from poornn import SPConv, functions

__all__=['RBM']

class RBM(ANN):
    '''
    Restricted Boltzmann Machine class.

    Attributes:
        :input_shape: tuple, (1, N1, N2 ...)
        :num_feature_hidden: int, number features in hidden layer.
    '''
    def __init__(self, input_shape, num_feature_hidden, itype='complex128'):
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

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return '<RBM> in[%s] hid[%s x %s]'%(self.input_shape, self.layers[0].output_shape)

    def __call__(self, config):
        return self.get_weight(config)

    @property
    def nsite(self):
        return np.prod(self.input_shape)

    def tovec(self,spaceconfig):  #poor designed interface.
        '''
        Get the state vector.

        \Psi(s,W)=\sum_{\{hi\}} e^{\sum_j a_j\sigma_j^z+\sum_i b_ih_i +\sum_{ij}W_{ij}h_i\sigma_j}
        '''
        configs=config=1-2*spaceconfig.ind2config(np.arange(spaceconfig.hndim))
        return np.array([self.get_weight(config_i) for config_i in configs])

    def get_weight(self,config,theta=None):
        '''
        Get the weight for specific configuration.

        Parameters:
            :config: 1darray,
            :theta: 1darray/None, table of hidden layer output: b+v.dot(W), intended to boost operation.

        Return:
            number,
        '''
        return self.forward(config)

