'''
runtime tools for monitoring programs.

testsuit take (optimizer, problem) as parameters.
'''

import numpy as np
import pdb, os
from matplotlib import pyplot as plt

from utils import analyse_exact
from plotlib import compare_wf, scatter_vec_phase

__all__ = ['Show_wf', 'Print_eng_with_exact',
        'show_wf', 'print_eng_with_exact', 'print_eng']

class Show_wf(object):
    '''show wave function, amplitude and phase.'''
    def __init__(self, D=0.8):
        self.D = D
        self.vv_pre = None

    def __call__(self,problem, optimizer):
        require_new_vv(problem, optimizer.n_iter)
        require_e0v0(problem)
        v0 = problem.cache['v0']

        num_iter = optimizer.n_iter
        vvi = 'vv-%s'%num_iter
        vv = problem.cache[vvi]

        plt.ion()
        fig=gcf()
        fig.set_figwidth(10,forward=True)
        fig.set_figheight(5,forward=True)
        fig.clear()

        plt.subplot(121)
        compare_wf(vv, v0)
        plt.subplot(122)
        scatter_vec_phase(vv, self.vv_pre)
        plt.xlim(-self.D,self.D)
        plt.ylim(-self.D,self.D)
        plt.pause(0.01)
        self.vv_pre = vv

class Print_eng_with_exact(object):
    '''print energy and compare it with exact value.'''
    def __init__(self, EG=None):
        self.EG = EG

    def __call__(self,problem,optimizer):
        nsite = problem.hamiltonian.nsite
        if self.EG is None:
            require_e0v0(problem)
            e0 = problem.cache['e0']
        else:
            e0 = self.EG
        ei = problem.cache['opq_vals'][0]  

        err=abs(e0-ei)/(abs(e0)+abs(ei))*2
        print('E/site = %s (%s), Error = %.4f%%'%(ei/nsite,e0/nsite,err*100))

class DumpNetwork(object):
    def __init__(self, folder='.', token='', step=1000):
        self.step = step
        self.folder = folder
        self.token = token

    def __call__(self, problem, optimizer):
        rbm = problem.rbm
        num_iter = optimizer.n_iter
        if num_iter%self.step!=0: return
        variables = rbm.get_variables()
        np.save(os.path.join(self.folder,'variables-%s%s.npy'%(self.token,num_iter)), variables)

def print_eng(problem,optimizer):
    '''print energy.'''
    ei = problem.cache['opq_vals'][0]  
    print('E/site = %s'%(ei/h.nsite,))


############### Requirements ###################
# `sampels` and `opq_vals` are cached by problem by default, other cached values must be regenerated!

def require_e0v0(problem, num_eng=1):
    if 'e0' not in problem.cache:
        H, e, v, configs = analyse_exact(problem.hamiltonian, num_eng=num_eng)
        problem.cache['e0'] = e
        problem.cache['v0'] = v
        problem.cache['H'] = H
        problem.cache['configs'] = configs
        return problem

def require_new_vv(problem, num_iter):
    vvi = 'vv-%s'%num_iter
    if vvi not in problem.cache:
        vv = problem.rbm.tovec(mag=problem.h.mag)
        problem.cache[vvi] = vv/np.linalg.norm(vv)

    # delete old one to avoid memory blow up.
    vvim = 'vv-%s'%(num_iter-1)
    if vvim in problem.cache:
        del(problem.cache[vvim])

############### predefined monitors ##################
show_wf = Show_wf()
print_eng_with_exact = Print_eng_with_exact()
