'''
Restricted Boltzmann Machine.
'''

from __future__ import division
import numpy as np
import pdb

from qstate import StateNN
from poornn.utils import typed_uniform
from poornn import SPConv, Linear, functions, ParallelNN, pfunctions, monitors, ANN
from poornn.checks import check_numdiff
from .layers import IsingRG2D, XLogcosh

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
    def __init__(self, input_shape, nonlinear_list, powerlist=None, num_features=[4,4,4], eta0=0.2, eta1=0.2, NP=1, NC=1, K=None,\
            itype='complex128',dtype0='complex128', dtype1='complex128', momentum=0., isym = False,
                    usesum=False,poly_order=10, do_BN=False,is_unitary=False, **kwargs):
        if K is None:
            K=np.prod(input_shape)
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

    def use_nonlinear(self,nonlinear):
        # non-linear function
        if nonlinear[-2:] == '_n':
            self.add_layer(functions.Normalize,axis=None,scale=1.0*np.sqrt(np.prod(self.layers[-1].output_shape)))
            self.add_layer(pfunctions.PMul,c=1.)
            nonlinear = nonlinear[:-2]

        if nonlinear=='none':
            pass
        elif nonlinear=='x^3':
            self.add_layer(functions.Power,order=3)
        elif nonlinear=='x^5':
            self.add_layer(functions.Power,order=5)
        elif nonlinear=='relu':
            self.add_layer(functions.ReLU)
        elif nonlinear=='sinh':
            self.add_layer(functions.Sinh)
        elif nonlinear=='softplus':
            self.add_layer(functions.SoftPlus)
        elif nonlinear=='tanh':
            self.add_layer(functions.Tanh)
        elif nonlinear=='sin':
            self.add_layer(functions.Sin)
        elif nonlinear=='tan':
            self.add_layer(functions.Tan)
        elif nonlinear=='log2cosh':
            self.add_layer(functions.Log2cosh)
        elif nonlinear=='logcosh':
            self.add_layer(functions.Logcosh)
        elif nonlinear=='ks_logcosh':
            from poornn import KeepSignFunc
            layer = functions.Logcosh(self.layers[-1].output_shape, 'float64')
            self.layers.append(KeepSignFunc(layer))
        elif nonlinear=='cos':
            self.add_layer(functions.Cos)
        elif nonlinear=='exp':
            self.add_layer(functions.Exp)
        elif nonlinear=='arctan':
            self.add_layer(functions.ArcTan)
        elif nonlinear=='IsingRG2D':
            self.add_layer(IsingRG2D)
        elif nonlinear=='XLogcosh':
            self.add_layer(XLogcosh)
        elif nonlinear=='ks_IsingRG2D':
            from poornn import KeepSignFunc
            layer = IsingRG2D(self.layers[-1].output_shape, 'float64')
            self.layers.append(KeepSignFunc(layer))
        elif nonlinear=='ks_x^1/3':
            from poornn import KeepSignFunc
            layer = functions.Power(self.layers[-1].output_shape, 'float64', order=1.0/3)
            self.layers.append(KeepSignFunc(layer))
        elif nonlinear=='gaussian':
            self.layers.append(pfunctions.Gaussian, params=[0j,1.],var_mask=[False,True])
        elif nonlinear=='real':
            self.add_layer(functions.Real)
        elif nonlinear[-2:]=='_r' and nonlinear[:-2] in pfunctions.Poly.kernel_dict:
            params = self.eta1*typed_uniform(self.layers[-1].otype, (self.poly_order,))
            #var_mask=np.array([False, True]*(self.poly_order//2)+[False]*(self.poly_order%2))
            #params[~var_mask] = 0
            #params[var_mask]*=np.sign(params[var_mask].real)  # positive real part
            self.add_layer(pfunctions.Poly, params=params, kernel=nonlinear[:-2], factorial_rescale=True, var_mask=None)
        elif nonlinear in pfunctions.Poly.kernel_dict:
            params = self.eta1*typed_uniform(self.layers[-1].otype, (self.poly_order,))
            #var_mask=np.array([False, True]*(self.poly_order//2)+[False]*(self.poly_order%2))
            #params[~var_mask] = 0
            #params[var_mask]*=np.sign(params[var_mask].real)  # positive real part
            self.add_layer(pfunctions.Poly, params=params, kernel=nonlinear, factorial_rescale=False, var_mask=None)
        else:
            raise Exception
