import numpy as np 
import scipy

from poornn import KeepSignFunc
from poornn.checks import check_numdiff
from poornn.functions import wrapfunc, Logcosh

__all__ = ['IsingRG2D']

IsingRG2D = wrapfunc(lambda x:0.375*Logcosh.forward(4*x),lambda xy,dy:1.5*scipy.tanh(4*xy[0])*dy,classname='IsingRG2D')
XLogcosh = wrapfunc(lambda x:x*Logcosh.forward(x),lambda xy,dy:(Logcosh.forward(xy[0])+xy[0]*scipy.tanh(xy[0]))*dy,classname='XLogcosh')
#assert(all(check_numdiff(IsingRG2D([5,5],'complex128'))))
