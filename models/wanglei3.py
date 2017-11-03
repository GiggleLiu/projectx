'''
Restricted Boltzmann Machine.
'''

from __future__ import division
import numpy as np
import pdb

from qstate import StateNN
from poornn.utils import typed_uniform
from poornn import SPConv, Linear, functions, ParallelNN, layers

__all__=['WangLei3']

class WangLei3(StateNN):
    '''
    Restricted Boltzmann Machine class.

    Attributes:
        :input_shape: tuple, (1, N1, N2 ...)
        :num_feature_hidden: int, number features in hidden layer.
    '''
    def __init__(self, input_shape, itype, powerlist, num_features=[12],fixbias=False,
            version='conv', stride=1, eta=0.2, usesum=False, nonlinear='x^3',momentum=0., poly_order=10, with_exp=False,
            factorial_rescale=False, nonlinear_mask = [False,False], **kwargs):
        self.num_features = num_features
        nsite=np.prod(input_shape)

        D = len(input_shape)
        ishape = (1,)+input_shape
        super(WangLei3, self).__init__(layers = [functions.Reshape(input_shape, itype=itype, output_shape=ishape)], **kwargs)

        NF = len(powerlist)
        plnn = ParallelNN(axis=1)
        for power in powerlist:
            plnn.layers.append(functions.ConvProd(input_shape = ishape, itype=itype,
                powers=power, boundary='P', strides=(stride,)*D))
        self.layers.append(plnn)
        if version=='linear':
            self.add_layer(functions.Reshape, output_shape=(np.prod(self.layers[-1].output_shape),))
            self.add_layer(Linear, weight=eta*typed_uniform(self.itype, (num_features[0], self.layers[-1].output_shape[-1])),
                    bias=(0 if fixbias else eta)*typed_uniform(self.itype, (num_features[0],)),var_mask=(1,0 if fixbias else 1))
        elif version=='conv':
            stride= 1
            imgsize = self.layers[-1].output_shape[-D:]
            self.add_layer(SPConv, weight=eta*typed_uniform(self.itype, (self.num_features[0], NF)+imgsize),
                    bias=(0 if fixbias else eta)*typed_uniform(self.itype, (num_features[0],)), boundary='P',
                    strides=(stride,)*D, var_mask=(1,0 if fixbias else 1))
            self.add_layer(functions.Reshape, output_shape=(num_features[0], np.prod(imgsize)//stride**D))

            if nonlinear=='x^3':
                self.add_layer(functions.Power,order=3)
            elif nonlinear=='x^5':
                self.add_layer(functions.Power,order=5)
            elif nonlinear=='relu':
                self.add_layer(functions.ReLU)
            elif nonlinear=='sinh':
                self.add_layer(functions.Sinh)
            elif nonlinear=='log2cosh':
                self.add_layer(functions.Log2cosh)
            elif nonlinear=='mobius':
                self.add_layer(layers.Mobius, params = np.array([0,1,1e20], dtype=itype),var_mask=[True,True,False])
            elif nonlinear=='softmax':
                self.add_layer(functions.SoftMax, axis=-1)
            elif nonlinear=='sin':
                self.add_layer(functions.Sin)
            elif nonlinear in layers.Poly.kernel_dict:
                self.add_layer(layers.Poly, params=eta*typed_uniform(itype, (poly_order,)), kernel=nonlinear, factorial_rescale=factorial_rescale)
            else:
                raise Exception
            #self.add_layer(functions.Filter, axes=(-1,), momentum=momentum)
            self.add_layer(functions.Mean, axis=-1)
            if nonlinear_mask[0]: 
                self.add_layer(functions.Sinh, params=eta*typed_uniform(itype, (poly_order,)), kernel='legendre', factorial_rescale=factorial_rescale)
                #self.add_layer(layers.Poly, params=eta*typed_uniform(itype, (poly_order,)), kernel='legendre', factorial_rescale=factorial_rescale)
        if version=='const-linear': 
            self.add_layer(Linear, weight=np.array([[-1,-1,1,1]],dtype=itype, order='F'),
                    bias=np.zeros((1,),dtype=itype),var_mask=(0,0))
        elif version=='linear' or version=='conv':
            if usesum:
                self.add_layer(functions.Mean, axis=-1)
            else:
                for i,(nfi, nfo) in enumerate(zip(num_features, num_features[1:]+[1])):
                    self.add_layer(Linear, weight=eta*typed_uniform(self.itype, (nfo, nfi)),
                            bias=(0 if fixbias else eta)*typed_uniform(self.itype, (nfo,)),var_mask=(1,0 if fixbias else 1))
                    #self.add_layer(layers.Poly, params=eta*typed_uniform(itype,(10,)))
        elif version=='rbm':
            pass
        else:
            raise ValueError('version %s not exist'%version)
        if with_exp:
            self.add_layer(functions.Exp)
        self.add_layer(functions.Reshape, output_shape=())
        if nonlinear_mask[1]: 
            self.add_layer(layers.Poly, params=eta*typed_uniform(itype, (poly_order,)), kernel='legendre', factorial_rescale=factorial_rescale)
