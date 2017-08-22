'''Random Boltzmann Machine Kernel for Monte Carlo.'''

import numpy as np
import pdb,time

from sampler.core import ConfigGenerator

__all__=['RBMConfigGenerator']

class RBMConfigGenerator(ConfigGenerator):
    '''
    Monte Carlo sampling core for Restricted Boltzmann Machine state.

    Attributes:
        :state: <RBM>,

        import torch
        :runtime: dict, runtime variables.
    '''
    def __init__(self, nflip, inverse_rate=0, initial_config=None):
        self.nflip=nflip
        self.state=None
        if hasattr(initial_config,'__iter__'):
            initial_config=np.asarray(initial_config)
        self.config=initial_config
        self.runtime={'ys': None, 'ys_p': None}
        self.inverse_rate=inverse_rate

    ########################################### Realize the interfaces ########################################
    def set_state(self,state):
        self.state=state
        if self.config is None:
            self.config=self.random_config()
        self.runtime['ys']=self.state.forward(self.config)
        self.runtime['ys_p']=None

    def random_config(self):
        rbm=self.state
        config=1-2*np.random.randint(0,2,rbm.nin)
        return config

    def fire(self):
        '''Fire a proposal.'''
        nsite=len(self.config)
        
        #generate a new config by flipping n spin
        if np.random.random() < self.inverse_rate:
            config_p = -config
        else:
            if self.nflip==2:  #for Heisenberg model.
                #flips=np.random.randint(0,nsite,2)       #why this code is wrong?
                #while flips[0]==flips[1]:
                    #flips=np.random.randint(0,nsite,2)
                upmask=self.config==1
                flips=np.random.randint(0,nsite/2,2)
                iflip0=np.where(upmask)[0][flips[0]]
                iflip1=np.where(~upmask)[0][flips[1]]
                flips=np.array([iflip0,iflip1])
            else:   #for transverse field Ising model
                iflip0=np.random.randint(nsite)
                flips=np.array([iflip0])

            #transfer probability is equal, pratio is equal to the probability ratio
            config_p=self.config.copy()
            config_p[flips]*=-1
        self.runtime['ys_p'] = self.state.forward(config_p)  #proposed ys.
        pop1=self.runtime['ys_p'][-1]/self.runtime['ys'][-1]
        return config_p,abs(pop1)**2

    def reject(self,*args,**kwargs):
        self.runtime['ys_p']=None

    def confirm(self,config_p,*args,**kwargs):
        #self.config[flips]*=-1
        self.config = config_p
        self.runtime['ys']=self.runtime['ys_p']
        self.runtime['ys_p']=None
