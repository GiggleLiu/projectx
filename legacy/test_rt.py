'''
Test Stochastic Reconfiguration with minimal model.
'''
from numpy import *
from numpy.testing import dec,assert_,assert_raises,assert_almost_equal,assert_allclose
from scipy.linalg import kron,eigh,norm
from matplotlib.pyplot import *
from profilehooks import profile
import sys,pdb,time
import scipy.sparse as sps
from os import path
sys.path.insert(0,'../')

from utils import sx,sy,sz, unpacknbits_pm, packnbits_pm
from spaceconfig import SpinSpaceConfig
from wanglei2 import WangLei2
from wanglei import WangLei
from models import *
from toymodel import ExactVMC
from sr import SR
from cgen import *
from sampler.vmc import *
from climin import RmsProp,GradientDescent,Adam
from probdef import ProbDef

from test_vmc import analyse_sampling

random.seed(2)

class ProbRTH(ProbDef):
    def __init__(self, nsite, periodic, model='AFH', num_feature=4):
        #Generate a model
        self.nsite,self.periodic=nsite,periodic
        self.model=model
        if model=='AFH':
            h=HeisenbergH(nsite=nsite,J1=1.,J1z=1.,periodic=periodic)
        elif model=='J1J2':
            h=HeisenbergH(nsite=nsite,J1=1.,J1z=1.,J2=0.2,J2z=0.2,periodic=periodic)
        elif model=='TFI':
            h=TFI(nsite=nsite,Jz=-4.,h=-1.,periodic=periodic)
        elif model=='AFH2D':
            N1=N2=int(sqrt(nsite))
            h=HeisenbergH2D(N1,N2,J=-1.,Jz=1.,periodic=periodic)
        else:
            raise ValueError()

        #Create a VMC sampling engine.
        cgen=RBMConfigGenerator(initial_config=[-1,1]*(nsite/2)+[1]*(nsite%2),nflip=2 if model=='AFH' or model=='J1J2' else 1)
        vmc=VMC(cgen,nbath=100*nsite,nsample=1000*nsite,nmeasure=nsite,sampling_method='metropolis')

        # Generate a random rbm and the corresponding vector v
        #rbm=WangLei2(input_shape=(nsite,),num_feature_hidden=num_feature, use_msr=False)
        rbm=WangLei(input_shape=(nsite,),num_feature_hidden=num_feature, dtype='complex128')

        ##################### choose an regulation method ##############################
        reg_params=('delta',{'lambda0':1e-4})
        #reg_params=('trunc',{'lambda0':0.2,'eps_trunc':1e-3})
        #reg_params=('carleo',{'lambda0':100,'b':0.9})
        #reg_params=('sd',{})
        #reg_params=('pinv',{})

        sr=SR(h, reg_params=reg_params, mode='r-theta' if hasattr(rbm,'thnn') else 'single-net', rtheta_training_ratio=30.)

        ##################### choose a optimization method with above gradients known ##############################
        #optimizer=RmsProp(wrt=rbm.get_variables(),fprime=self.compute_gradient,step_rate=1e-3,decay=0.9,momentum=0.)
        #optimizer=Adam(wrt=rbm.get_variables(),fprime=self.compute_gradient,step_rate=1e-2)
        optimizer=GradientDescent(wrt=rbm.get_variables(),fprime=self.compute_gradient,step_rate=2e-2,momentum=0.)

        super(ProbRTH,self).__init__(h, rbm, vmc, sr, optimizer)
 
    def compute_gradient(self, x):
        '''
        Compute the gradient.
        '''
        vmc, rbm, sr = self.vmc, self.rbm, self.sr

        # update RBM
        rbm.set_variables(x)

        # generate samples probabilities given by this new RBM
        samples = vmc.generate_samples(rbm)

        # assign signs to samples
        #rbm.thnn.render_theta(samples)

        # perform measurements on operators that needed by stochastic gradient generator.
        opq_vals = sr.opq.eval_on_samples(samples, rbm, update_samples=False)

        # get gradient from observables
        gradient = sr.get_gradient(opq_vals)

        self.cache['opq_vals'] = opq_vals
        self.cache['samples'] = samples
        return gradient

if __name__=='__main__':
    t=ProbRTH(nsite=6,periodic=True,model='AFH')
    t.run(compare_to_exact=True, print_var_diff = False, check_sample=False, plot_wf=True)
