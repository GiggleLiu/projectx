'''
Test Stochastic Reconfiguration with minimal model.
'''
from numpy import *
from numpy.testing import dec,assert_,assert_raises,assert_almost_equal,assert_allclose
from scipy.linalg import kron,eigh,norm
from matplotlib.pyplot import *
import sys,pdb,time
from os import path
sys.path.insert(0,'../')

from utils import sx,sy,sz
from spaceconfig import SpinSpaceConfig
from wanglei import *
from models import *
from toymodel import ExactVMC
from sr import *
from cgen import *
from sampler.vmc import *
from group import TIGroup,NoGroup
from climin import RmsProp,GradientDescent,Adam
from poornn.checks import check_numdiff
from probdef import ProbDef

from test_vmc import analyse_sampling

random.seed(2)

class SRTest(ProbDef):
    def __init__(self,nsite,periodic,model='AFH'):
        #Generate a model
        self.nsite,self.periodic=nsite,periodic
        self.model=model
        if model=='AFH':
            self.h=HeisenbergH(nsite=nsite,J1=-1.,J1z=1.,periodic=periodic)
        elif model=='J1J2':
            self.h=HeisenbergH(nsite=nsite,J1=1.,J1z=1.,J2=0.2,J2z=0.2,periodic=periodic)
        elif model=='TFI':
            self.h=TFI(nsite=nsite,Jz=-4.,h=-1.,periodic=periodic)
        elif model=='AFH2D':
            N1=N2=int(sqrt(nsite))
            self.h=HeisenbergH2D(N1,N2,J=-1.,Jz=1.,periodic=periodic)
        else:
            raise ValueError()
        #Hilbert Space configuration, # of site, # of spin.
        self.scfg=SpinSpaceConfig([nsite,2])

        #A disguished VMC engine, used for tests.
        self.fv=ExactVMC(self.h)

        #Create a VMC sampling engine.
        cgen=RBMConfigGenerator(initial_config=[-1,1]*(nsite/2)+[1]*(nsite%2),nflip=2 if model=='AFH' or model=='J1J2' else 1)
        self.vmc=VMC(cgen,nbath=200*nsite,nsample=1000*nsite,nmeasure=nsite,sampling_method='metropolis')

        ##################### choose an regulation method ##############################
        reg_params=('delta',{'lambda0':1e-4})
        #reg_params=('trunc',{'lambda0':0.2,'eps_trunc':1e-3})
        #reg_params=('carleo',{'lambda0':100,'b':0.9})
        #reg_params=('sd',{})
        #reg_params=('pinv',{})

        nfeature=4
        self.rbm=WangLei(input_shape=(self.nsite,),num_feature_hidden=nfeature, dtype='float64')
        self.sr=SR(self.h, reg_params=reg_params)
        self.cache = {}

        ##################### choose a optimization method with above gradients known ##############################
        #optimizer=RmsProp(wrt=self.rbm.get_variables(),fprime=sr.compute_gradient,step_rate=1e-3,decay=0.9,momentum=0.)
        #optimizer=Adam(wrt=self.rbm.get_variables(),fprime=sr.compute_gradient,step_rate=1e-2)
        self.optimizer=GradientDescent(wrt=self.rbm.get_variables(),fprime=self.compute_gradient,step_rate=2e-2,momentum=0.)

    def test_sr(self,fakevmc=False):
        do_check_numdiff = False
        plot_wf_distri = False
        print_var_diff = False
        plot_wf = False
        compare_to_exact = True

        #generate a random rbm and the corresponding vector v
        rbm = self.rbm
        sr = self.sr

        if do_check_numdiff:
            diff=check_numdiff(self.rbm, num_check=500, eta=1e-3)
            for layer in self.rbm.layers:
                print 'Testing %s'%layer
                diff=check_numdiff(layer, num_check=10, eta=1e-3)

        # Exact Results
        if compare_to_exact:
            H0, e0, v0, configs = self.analyse_exact(do_printsign=self.nsite<=6)
            if plot_wf_distri:
                v0_ = sort(abs(v0))[::-1]
                plot(v0_)
                pdb.set_trace()

       
        #optimization variables(b and W) are dumped to an 1darray.
        if print_var_diff: 
            arr_old=self.rbm.get_variables()
        el=[] # to store error

        # plot
        if (plot_wf and compare_to_exact):
            ion()
            fig=figure(figsize=(10,5))

        print '\nRunning 0-th Iteration.'
        for info in self.optimizer:
            #rbm state -> vector state, and then compare it with exact ground state vector
            ei=self.cache['opq_vals'][0]

            if compare_to_exact:
                if plot_wf:
                    vv = array([self.rbm.forward(config)[-1]*sr._sign_classifier(config) for config in configs])
                    vv = vv/linalg.norm(vv)
                    if vv.dot(v0)<0: vv=-vv
                    print('|<Psi_0|Psi>|^2 = %s'%abs(v0.dot(vv))**2)
                    cla()
                    title('Wave Function')
                    plot(v0, lw=1, color='k')
                    plot(vv, lw=1, color='r')
                    legend([r'$\Psi_0$', r'$\Psi$'])
                    fig.canvas.draw()
                    pause(0.2)
 
                err=abs(e0-ei)/(abs(e0)+abs(ei))*2
                print('E/site = %s (%s), Error = %.4f%%'%(ei/self.nsite,e0/self.nsite,err*100))
            else:
                print('E/site = %s'%(ei/self.nsite,))
            el.append(ei)

            if print_var_diff:
                #see to what extent, variables are changed with respect to those in last iteration.
                arr=self.rbm.get_variables()
                print 'diff rate = %s(norm=%s)'%(norm(arr-arr_old)/norm(arr_old),norm(arr_old))
                arr_old=arr

            if info['n_iter']>=50:
                self.vmc.nsample=1000*self.nsite
            if info['n_iter']>=800:
                if plot_wf: savefig('data/WF_J2%s_N%s.pdf'%(self.h.J2,self.nsite))
                break
            print '\nRunning %s-th Iteration.'%(info['n_iter']+1)

        savetxt('data/el-%s%s.dat'%(self.nsite,'p' if self.periodic else 'o'),el)
        pdb.set_trace()

    def compute_gradient(self, x):
        '''
        Compute the gradient.
        '''
        vmc, rbm, sr = self.vmc, self.rbm, self.sr

        # update RBM
        rbm.set_variables(x)

        # generate samples probabilities given by this new RBM
        random.seed(99)
        samples = vmc.generate_samples(rbm)

        # perform measurements on operators that needed by stochastic gradient generator.
        opq_vals = sr.opq.eval_on_samples(samples, rbm, update_samples=True)

        # get gradient from observables
        gradient = sr.get_gradient(opq_vals)

        self.cache['opq_vals'] = opq_vals
        self.cache['samples'] = samples
        return gradient

def show_err_sr(nsite):
    from matplotlib.pyplot import plot,ion
    ion()
    fig=figure(figsize=(5,4))
    for b,c in zip(['p','o'],['gray','k']):
        f='data/err-%s%s.dat'%(nsite,b)
        f0='data/err0-%s%s.dat'%(nsite,b)
        el=loadtxt(f)
        el0=loadtxt(f0)
        plot(el0,lw=2,color=c)
        plot(el,lw=2,color=c,ls='--')
    xlabel('iteration',fontsize=16)
    ylabel(r'$Err=\frac{|E-E_0|}{|E|+|E_0|}$',fontsize=16)
    #ylabel(r'$1-\|\left\langle\psi|\tilde{\psi}\right\rangle\|_2$',fontsize=16)
    ylim(1e-8,1)
    yscale('log')
    legend(['Exact/Periodic','VMC/Periodic','Exact/Open','VMC/Open'],loc=3)
    tight_layout()
    pdb.set_trace()
    savefig('data/err-%s.pdf'%nsite)

if __name__=='__main__':
    t=SRTest(nsite=16,periodic=True,model='J1J2')
    #t.test_sr()
    t.run(compare_to_exact=True, plot_wf=True)
    #show_err_sr(nsite=4)
