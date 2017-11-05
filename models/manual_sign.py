'''
Restricted Boltzmann Machine.
'''

from __future__ import division
import numpy as np
import pdb

from poornn.utils import typed_uniform
from poornn import SPConv, Linear, functions, ParallelNN, pfunctions, monitors, ANN
from poornn.checks import check_numdiff
from .base import StateNN
from qstate.core.utils import packnbits_pm

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
    '''
    def __init__(self, sign_func, input_shape, nonlinear_list, powerlist=None, num_features=[4,4,4], eta0=0.2, eta1=0.2, NP=1, NC=1, K=None,\
            itype='complex128',dtype0='complex128', dtype1='complex128', momentum=0., isym = False,
                    usesum=False,poly_order=10, do_BN=False,is_unitary=False, **kwargs):
        if K is None:
            K=np.prod(input_shape)
        self._sign_func = sign_func
        self.num_features = num_features
        self.do_BN = do_BN
        self.poly_order = poly_order
        self.eta0, self.eta1 = eta0, eta1
        self.isym = isym
        nsite=np.prod(input_shape)
        D = len(input_shape)
        ishape = input_shape
        def _is_unitary(inl):
            if is_unitary is True:
                return True
            elif is_unitary is False:
                return False
            else:
                return is_unitary[inl]

        super(WangLei6, self).__init__(**kwargs)

        # preprocessing
        if powerlist is not None:
            plnn = ParallelNN(axis=0)
            for power in powerlist:
                plnn.layers.append(functions.ConvProd(ishape, itype, powers=power, boundary='P', strides=(1,)*D))
                if isym:
                    plnn.layers.append(ANN(layers=[functions.Reverse(ishape, itype, axis=-1), 
                        functions.ConvProd(ishape, itype, powers=power, boundary='P', strides=(1,)*D)]))
            self.layers.append(plnn)
            nfo = len(plnn.layers)
        else:
            nfo = 1

        inl = -1
        # product layers.
        dtype = dtype0
        eta=eta0
        if NP!=0: self.add_layer(functions.Log,otype='complex128')
        for inl in range(NP):
            nfi, nfo = nfo, num_features[inl]
            self.add_layer(SPConv, weight=eta*typed_uniform(dtype, (nfo, nfi)+(K,)*D),
                    bias=eta*typed_uniform(dtype, (nfo,)), boundary='P', strides=(1,)*D,is_unitary=_is_unitary(inl))
            self.use_nonlinear(nonlinear_list[inl])
        if NP!=0: self.add_layer(functions.Exp)

        # convolution layers.
        eta=eta1
        dtype = self.layers[-1].otype
        for nfi, nfo in zip([nfo]+num_features[NP:NP+NC-1], num_features[NP:NP+NC]):
            inl = inl+1
            self.add_layer(SPConv, weight=eta*typed_uniform(dtype, (nfo, nfi)+input_shape),
                    bias=eta*typed_uniform(dtype, (nfo,)), boundary='P', strides=(1,)*D,is_unitary=_is_unitary(inl))
            self.use_nonlinear(nonlinear_list[inl])
        self.add_layer(functions.Filter, axes=(-1,), momentum=momentum)
        inl=inl+1
        self.use_nonlinear(nonlinear_list[inl])

        # linear layers.
        if usesum:
            self.add_layer(functions.Mean, axis=-1)
            inl=inl+1
            self.use_nonlinear(nonlinear_list[inl])
        else:
            for i,(nfi, nfo) in enumerate(zip(num_features[NP+NC-1:], num_features[NP+NC:]+[1])):
                self.add_layer(Linear, weight=eta*typed_uniform(dtype, (nfo, nfi)),
                        bias=eta*typed_uniform(dtype, (nfo,)),var_mask=(1,1),is_unitary=_is_unitary(inl))
                if do_BN:
                    self.add_layer(functions.BatchNorm, axis=None, label='BN-%s'%i)
                    self.add_layer(pfunctions.Poly, params=np.array([0,1.],dtype=dtype1), kernel='polynomial', factorial_rescale=True)
                inl=inl+1
                self.use_nonlinear(nonlinear_list[inl])
        print(check_numdiff(self))

    def forward(self,x,**kwargs):
        return super(WangLei6, self).forward(x, **kwargs)*self.get_sign(x)

    def backward(self, xy, dy, **kwargs):
        x, y = xy
        xy = x, y/self.get_sign(x)
        return super(WangLei6, self).backward(xy, dy/self.get_sign(x), **kwargs)

    def get_sign(self,x):
        return self._sign_func(x)
