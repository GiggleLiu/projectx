'''
Toy r-theta model
'''

import numpy as np
import numbers,pdb
from scipy.special import expit

from poornn.utils import typed_randn
from poornn import SPConv, Linear, functions
from ..classifier.signnn import PSNN
from statenn import StateNN

__all__=['ToyRTH']

class ToyRTH(StateNN):
    '''
    Toy r-theta network with amplitude fixed.

    Attributes:
        :use_msr: use marshall sign rule.
    '''
    def __init__(self, input_shape, use_msr=False, theta_period=2):
        self.use_msr = use_msr
        itype = 'float64'
        nsite=np.prod(input_shape)
        eta=0.1
        self.vec = eta*typed_randn(itype, input_shape)

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
        return self.thnn.get_variables()

    def set_variables(self, v):
        self.thnn.set_variables(v)

    @property
    def num_variables(self):
        return self.thnn.num_variables

    def forward(self, x):
        return [self.vec[packnbits_pm(x)]]

    def backward(self, dy):
        dw = dy*np.ones()
        return dw, dx
