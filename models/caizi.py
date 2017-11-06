'''
Restricted Boltzmann Machine.
'''

from __future__ import division
import numpy as np
import pdb
from poornn.utils import typed_randn
from poornn.checks import check_numdiff
from poornn import SPConv, Linear, functions, ANN, ParallelNN

from .base import StateNN

__all__=['CaiZi']

class CaiZi(StateNN):
    '''
    Restricted Boltzmann Machine class.

    Attributes:
        :input_shape: tuple, (N1, N2 ...)
        :num_feature_hidden: int, number features in hidden layer.
    '''
    def __init__(self, input_shape, num_features1=[12], num_features2=[],
            itype='float64',version='basic', eta=0.2, use_conv=False):
        self.num_features1, self.num_features2 = num_features1, num_features2
        nsite=np.prod(input_shape)
        super(CaiZi, self).__init__()

        # create amplitude network
        net1 = ANN(layers=[functions.Reshape(input_shape,itype=itype,output_shape=input_shape if not use_conv else ((1,)+input_shape))])
        for i,(nfi, nfo) in enumerate(zip([np.prod(input_shape)]+num_features1, num_features1+[1])):
            if use_conv and i==0:
                net1.add_layer(SPConv, weight=eta*typed_randn(self.itype, (nfo, 1, nsite)),
                        bias=eta*typed_randn(self.itype, (nfo,)))
                net1.add_layer(functions.Transpose, axes=(1,0))
            else:
                net1.add_layer(Linear, weight=eta*typed_randn(self.itype, (nfo, nfi)),
                        bias=eta*typed_randn(self.itype, (nfo,)))
            if version=='basic':
                net1.add_layer(functions.Tanh)
            elif version=='sigmoid':
                net1.add_layer(functions.Sigmoid)
            else:
                raise
        if use_conv:
            net1.add_layer(functions.Sum, axis=0)
        net1.add_layer(functions.Reshape, output_shape=())

        # create sign network
        net2 = ANN(layers=[functions.Reshape(input_shape,itype=itype,output_shape=input_shape if not use_conv else ((1,)+input_shape))])
        for i,(nfi, nfo) in enumerate(zip([np.prod(input_shape)]+num_features2, num_features2+[1])):
            if use_conv and i==0 and False:
                net1.add_layer(SPConv, weight=eta*typed_randn(self.itype, (nfo, nfi, nsite)),
                        bias=eta*typed_randn(self.itype, (nfo,)))
                net1.add_layer(functions.Transpose, axes=(1,0))
            else:
                net2.add_layer(Linear, weight=eta*typed_randn(self.itype, (nfo, nfi)),
                        bias=eta*typed_randn(self.itype, (nfo,)))
            net2.add_layer(functions.Mul, alpha=np.pi)
            net2.add_layer(functions.Cos)
        if use_conv:
            net2.add_layer(functions.Sum, axis=0)
        net2.add_layer(functions.Reshape, output_shape=())

        # construct whole network
        self.layers.append(ParallelNN(layers = [net1,net2]))
        self.add_layer(functions.Prod, axis=0)
        print(check_numdiff(self))
