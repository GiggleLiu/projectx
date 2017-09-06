'''
Test Stochastic Reconfiguration with minimal model.
'''
from __future__ import division
import pdb
import os,sys
import numpy as np

from qstate.sampler import SR, SpinConfigGenerator, VMC, hamiltonians
from climin import RmsProp,GradientDescent,Adam

class ProbDef(object):
    def __init__(self, hamiltonian, rbm, vmc, sr, num_vmc_run, num_vmc_sample):
        self.hamiltonian, self.rbm, self.vmc, self.sr = hamiltonian, rbm, vmc, sr
        self.cache = {}
        self.num_vmc_run = num_vmc_run
        self.num_vmc_sample = num_vmc_sample

    def compute_gradient(self, x):
        '''
        Compute the gradient.
        '''
        vmc, rbm, sr = self.vmc, self.rbm, self.sr

        # update RBM
        rbm.set_variables(x)

        # generate samples probabilities given by this new RBM
        samples = vmc.mpi_generate_samples(self.num_vmc_run, rbm, num_sample=self.num_vmc_sample)

        # assign signs to samples
        #rbm.thnn.render_theta(samples)

        # perform measurements on operators that needed by stochastic gradient generator.
        opq_vals = sr.opq.mpi_eval_on_samples(samples, rbm, update_samples=False)

        # get gradient from observables
        gradient = sr.get_gradient(opq_vals)

        self.cache['opq_vals'] = opq_vals
        self.cache['samples'] = samples
        return gradient

class ModelProbDef(ProbDef):
    '''
    Attributes:
        :reg_method: str, regulation method,

            * sd, steepest descent.
            * delta, add a small number to the diagonal part of S matrix.
            * trunc, truncation regulation approach for SR.
            * carleo, the carleo's version, buggy!.
            * pinv, use pseudo-inverse.
    '''
    def __init__(self, hamiltonian, rbm, reg_method, num_vmc_run=1, num_vmc_sample=1000, sr_layerwise=True):
        self.reg_method = reg_method
        #Create a VMC sampling engine.
        cgen=SpinConfigGenerator(initial_config=[-1,1]*(hamiltonian.nsite//2)+\
                [1]*(hamiltonian.nsite%2),nflip=2 if hamiltonian.mag is not None else 1, inverse_rate=0.05)
        vmc=VMC(cgen, nbath=200*hamiltonian.nsite, measure_step=hamiltonian.nsite,sampling_method='metropolis',iprint=0)

        ##################### choose an regulation method ##############################
        if reg_method == 'delta':
            reg_params=('delta',{'lambda0':1e-4})
        elif reg_method == 'trunc':
            reg_params=('trunc',{'lambda0':0.2,'eps_trunc':1e-3})
        elif reg_method == 'carleo':
            reg_params=('carleo',{'lambda0':100,'b':0.9})
        elif reg_method == 'sd':
            reg_params=('sd',{})
        elif reg_method == 'pinv':
            reg_params=('pinv',{})
        else:
            raise ValueError('Can not load predefined regulation method %s'%reg_method)

        sr=SR(hamiltonian, reg_params=reg_params, state=rbm, rtheta_training_ratio=[1.,30.], layer_wise=sr_layerwise)
        super(ModelProbDef,self).__init__(hamiltonian, rbm, vmc, sr, num_vmc_run=num_vmc_run, num_vmc_sample=num_vmc_sample)


def load_hamiltonian(model, size, periodic=True, **kwargs):
    J1=kwargs.get('J1',1.0)
    J1z=kwargs.get('J1z')
    if J1z is None: J1z=J1
    J2=kwargs.get('J2',0)
    J2z=kwargs.get('J2z')
    if J2z is None: J2z=J2
    h=kwargs.get('h',0)
    if model=='AFH':
        if len(size)==1:
            h=hamiltonians.HeisenbergH(nsite=size[0],J1=J1,J1z=J1z,periodic=periodic, mag=0)
        else:
            h=hamiltonians.HeisenbergH2D(size[0], size[1],J1=J1,J1z=J1z,periodic=periodic, mag=0)
    elif model=='J1J2':
        if len(size)==1:
            h=hamiltonians.HeisenbergH(nsite=size[0],J1=J1,J1z=J1z,J2=J2,J2z=J2,periodic=periodic, mag=0)
        else:
            h=hamiltonians.HeisenbergH2D(size[0], size[1],J1=J1,J1z=J1z,J2=J2,J2z=J2z,periodic=periodic, mag=0)
    elif model=='TFI':
        h=hamiltonians.TFI(nsite=size[0],Jz=J1z,h=h,periodic=periodic, mag=None)
    else:
        raise ValueError()
    return h


def load_config(config_file):
    from configobj import ConfigObj
    from validate import Validator

    #read config
    #specfile=os.path.join(os.path.dirname(__file__),'config-spec.ini')
    specfile=os.path.join(os.path.dirname(__file__),'config-spec.ini')
    config=ConfigObj(config_file,configspec=specfile,stringify=True)
    validator = Validator()
    result = config.validate(validator,preserve_errors=True)
    if result!=True:
        raise ValueError('Configuration Error! %s'%result)
    return config

def pconfig(config, rbm):
    '''
    Config a problem and optimizer.
    '''
    # hamiltonian
    hconfig = config['hamiltonian']
    hamiltonian = load_hamiltonian(**hconfig)

    # cgen
    cgenconfig = config['cgen']
    cgen=SpinConfigGenerator(initial_config=random_config(num_spin=hamiltonian.nsite, mag=cgenconfig['mag']), \
            nflip=cgenconfig['num_flip'], inverse_rate=cgenconfig['inverse_rate'])


    # vmc
    vmcconfig = config['vmc']
    vmc=VMC(cgen, nbath=vmcconfig['num_bath_sweep']*hamiltonian.nsite, \
            measure_step=vmcconfig['num_sweep_per_sample']*hamiltonian.nsite,\
            sampling_method=vmcconfig['accept_method'], iprint=vmcconfig['iprint'])

    # sr
    srconfig = config['sr']
    reg_method = srconfig['reg_method']
    lambda0 = srconfig['lambda']
    if reg_method == 'delta':
        reg_params=('delta',{'lambda0':lambda0})
    elif reg_method == 'trunc':
        reg_params=('trunc',{'lambda0':lambda0,'eps_trunc':srconfig['eps_trunc']})
    elif reg_method == 'carleo':
        raise NotImplementedError()
        reg_params=('carleo',{'lambda0':lambda0,'b':0.9})
    elif reg_method == 'sd':
        reg_params=('sd',{})
    elif reg_method == 'pinv':
        reg_params=('pinv',{})
    else:
        raise ValueError('Can not load predefined regulation method %s'%reg_method)

    sr=SR(hamiltonian, reg_params=reg_params, state=rbm, \
            rtheta_training_ratio=srconfig['rtheta_training_ratio'],\
            layer_wise=srconfig['sr_layerwise'])

    # optimize
    prob = ProbDef(hamiltonian, rbm, vmc, sr, \
            num_vmc_run=vmcconfig['num_vmc_run'], num_vmc_sample=vmcconfig['num_sample'])
    optimizeconfig = config['optimize']
    optimizer = get_optimizer(rbm.get_variables(), prob.compute_gradient, **optimizeconfig)
    return optimizer, prob


def get_optimizer(wrt, fprime, optimize_method, step_rate, momentum=0.0, decay=0.9, **kwargs):
    '''Get an optimizer.'''
    if optimize_method == 'rmsprop':
        optimizer=RmsProp(wrt=wrt, fprime=fprime,step_rate=step_rate, decay=decay, momentum=momentum)
    elif optimize_method == 'adam':
        optimizer=Adam(wrt=wrt,fprime=fprime,step_rate=step_rate)
    elif optimize_method == 'gd':
        optimizer=GradientDescent(wrt=wrt,fprime=fprime,step_rate=step_rate,momentum=momentum)
    else:
        raise ValueError('Can not load predefined optimization method %s'%optimize_method)
    return optimizer

def random_config(num_spin, mag=None):
    # generate a config
    config=1-2*np.random.randint(0,2,num_spin)
    if mag is None: return config

    if num_spin%2 != mag%2:
        raise ValueError('Parity of mag are not equal to number of spins!')
    if mag<-num_spin or mag>num_spin:
        raise ValueError('Mag greater than number of spins!')

    if config.sum()>mag:
        config*=-1
    while(config.sum()<mag):
        # pick a random position and flip
        ispin = np.random.randint(num_spin)
        config[ispin]=1
    return config

if __name__=='__main__':
    if len(sys.argv)>1:
        config_file=sys.argv[1]
    else:
        config_file='config-sample.ini'
    from models.wanglei4 import WangLei4
    size = (4,4)
    J2 = 0.8
    h = load_hamiltonian('J1J2', size=size, J2=J2)
    rbm = WangLei4(input_shape=size,NF=8, K=3,num_features=[8], version='conv', dtype='complex128')
    config = load_config(config_file)
