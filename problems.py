'''
Test Stochastic Reconfiguration with minimal model.
'''
import pdb

from qstate.sampler import SR, SpinConfigGenerator, VMC, hamiltonians
from climin import RmsProp,GradientDescent,Adam

class ProbDef(object):
    def __init__(self, hamiltonian, rbm, vmc, sr, optimizer, num_vmc_run, num_vmc_sample):
        self.hamiltonian, self.rbm, self.vmc, self.sr, self.optimizer = hamiltonian, rbm, vmc, sr, optimizer
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
    def __init__(self, hamiltonian, rbm, reg_method, optimize_method, step_rate, num_vmc_run=1, num_vmc_sample=1000):
        self.reg_method = reg_method
        #Create a VMC sampling engine.
        cgen=SpinConfigGenerator(initial_config=[-1,1]*(hamiltonian.nsite/2)+\
                [1]*(hamiltonian.nsite%2),nflip=2 if hamiltonian.mag is not None else 1)
        vmc=VMC(cgen, nbath=200*hamiltonian.nsite, measure_step=hamiltonian.nsite,sampling_method='metropolis')

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

        sr=SR(hamiltonian, reg_params=reg_params, mode='r-theta' if hasattr(rbm,'thnn') else 'single-net', rtheta_training_ratio=30.)

        ##################### choose a optimization method with above gradients known ##############################
        if optimize_method == 'rmsprop':
            optimizer=RmsProp(wrt=rbm.get_variables(),fprime=self.compute_gradient,step_rate=step_rate,decay=0.9,momentum=0.)
        elif optimize_method == 'adam':
            optimizer=Adam(wrt=rbm.get_variables(),fprime=self.compute_gradient,step_rate=step_rate)
        elif optimize_method == 'gd':
            optimizer=GradientDescent(wrt=rbm.get_variables(),fprime=self.compute_gradient,step_rate=step_rate,momentum=0.)
        else:
            raise ValueError('Can not load predefined optimization method %s'%optimize_method)

        super(ModelProbDef,self).__init__(hamiltonian, rbm, vmc, sr, optimizer, num_vmc_run=num_vmc_run, num_vmc_sample=num_vmc_sample)


def load_hamiltonian(model, size, periodic=True, **kwargs):
    J1=kwargs.get('J1',1.0)
    J1z=kwargs.get('J1z',J1)
    J2=kwargs.get('J2',0)
    J2z=kwargs.get('J2z',J2)
    h=kwargs.get('h',0)
    if model=='AFH':
        h=hamiltonians.HeisenbergH(nsite=size[0],J1=J1,J1z=J1z,periodic=periodic, mag=0)
    elif model=='J1J2':
        h=hamiltonians.HeisenbergH(nsite=size[0],J1=J1,J1z=J1z,J2=J2,J2z=J2,periodic=periodic, mag=0)
    elif model=='TFI':
        h=hamiltonians.TFI(nsite=size[0],Jz=J1z,h=h,periodic=periodic, mag=None)
    elif model=='AFH2D':
        h=hamiltonians.HeisenbergH2D(size[0], size[1],J1=J1,J1z=J1z,periodic=periodic, mag=0)
    elif model=='J1J22D':
        h=hamiltonians.HeisenbergH2D(size[0], size[1],J1=J1,J1z=J1z,J2=J2,J2z=J2z,periodic=periodic, mag=0)
    else:
        raise ValueError()
    return h
