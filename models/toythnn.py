import numpy as np
import pdb

from qstate.classifier import ThetaNN
from qstate.core.utils import packnbits_pm

class ToyTHNN(ThetaNN):
    '''
    Periodic NN to determine sign.

    Attributes:
        :input_shape: tuple, (N1, N2 ...)
        :num_feature_hidden: int, number features in hidden layer.
        :output_mode: 'sign'/'loss'/'theta'.
    '''
    def __init__(self, h, use_msr=False):
        self.h = h
        self.dtype='float64'
        self.vec = 2*np.pi*np.random.random(len(h.configs))

    def get_variables(self):
        return self.vec

    def set_variables(self, v, mode='set'):
        if mode=='set':
            self.vec[:]=v
        elif mode=='add':
            self.vec[:]+=v
        else:
            raise ValueError()

    @property
    def num_variables(self):
        nv = self.vec.shape[0]
        return nv

    @property
    def input_shape(self):
        return self.h.size

    def forward(self, x):
        return [x, self.vec[self.h.config_indexer[packnbits_pm(x)]]]

    def backward(self, ys, dy=None):
        if dy is None:
            dy = np.array(1,dtype=self.dtype)
        dw = np.zeros_like(self.vec)
        dw[self.h.config_indexer[packnbits_pm(ys[0])]]=dy
        return dw, None


