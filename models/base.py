'''
Basic State Ansatz
'''

import numpy as np
from poornn.nets import ANN
from qstate.core.utils import unpacknbits_pm
from poornn import linears
from .layers import IsingRG2D, XLogcosh

__all__=['StateNN']

class StateNN(ANN):
    '''
    Args:
        soften_gradient (bool): use a softer trainig method (when setting variables).
        cc (bool): accelerate the first layer.
    '''
    def __init__(self, *args, cc=False, soften_gradient=False, **kwargs):
        super(StateNN, self).__init__(*args, **kwargs)
        self.soften_gradient = soften_gradient
        self.cc = cc

    def __call__(self, config):
        return self.get_weight(config)

    @property
    def nsite(self):
        return np.prod(self.input_shape)

    def update(self, locs, dx, y0, data_cache=None, **kwargs):
        '''
        Feed input to this feed forward network.

        Args:
            locs (2darray): different positions in input.
            dx (ndarray): difference amount of value in input.
            data_cache (dict|None, default=None): a dict used to collect datas.

        Note:
            :data:`data_cache` should be pass to this method if you are about \
to call a subsequent :meth:`backward` method, \
because backward need :data:`data_cache`.

            :data:`self.uuid` is used as the key to store \
run-time output of layers in this network.
            :data:`data_cache[self.uuid]` is a list with contents \
outputs in each layers generate in this forward run.

        Returns:
            list: output in each layer.
        '''
        # find and cc first linear layer.
        offset = 0
        for layer in self.layers:
            if hasattr(layer, 'forward_cc'):
                x = layer.forward_cc(locs, dx, y0, data_cache=data_cache, **kwargs)
                break

        # continue training
        ys = []
        for layer in self.layers[offset:]:
            x = layer.forward(x, data_cache=data_cache, **kwargs)
            ys.append(x)
            if isinstance(x, list):
                x = x[-1]
        if data_cache is not None:
            data_cache[self.uuid] = ys
        return x

    def tovec(self, mag=None):
        '''
        Get the state vector.

        \Psi(s,W)=\sum_{\{hi\}} e^{\sum_j a_j\sigma_j^z+\sum_i b_ih_i +\sum_{ij}W_{ij}h_i\sigma_j}
        '''
        configs=unpacknbits_pm(np.arange(2**self.nsite), self.nsite)
        if mag is not None:
            configs = configs[configs.sum(axis=1)==mag]
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
        return self.forward(config)*self.get_sign(config)

    def get_sign(self, config, **kwargs):
        return 1

    def backward(self, xy, *args, **kwargs):
        dw, dx = super(StateNN, self).backward(xy, *args, **kwargs)
        if self.soften_gradient:
            x = self.get_variables()
            dw*=np.exp(-2*x)
        return dw, dx

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

def handle_cc(layer, locs, dx):
    '''
    handler perturbation acceleration.
    '''
    if hasattr(layers, 'cc_'):
        pass
