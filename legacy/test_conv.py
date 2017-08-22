'''
Test Stochastic Reconfiguration with minimal model.
'''
from numpy import *
from numpy.testing import dec,assert_,assert_raises,assert_almost_equal,assert_allclose
from scipy.linalg import kron,eigh,norm
from matplotlib.pyplot import *
import sys,pdb,time
from os import path
from profilehooks import profile
sys.path.insert(0,'../')

from utils import sx,sy,sz
from spaceconfig import SpinSpaceConfig
from convansatz import ConvWF
from models import *
from toymodel import ExactVMC
from sr import *
from cgen import *
from sampler.vmc import *
from group import TIGroup,NoGroup
from climin import RmsProp,GradientDescent,Adam
from poornn.checks import check_numdiff

from test_vmc import analyse_sampling

random.seed(2)

class SRTest(object):
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
            self.h=HeisenbergH2D(N1,N2,J1=-1.,J1z=1.,periodic=periodic)
        elif model=='J1J22D':
            N1=N2=int(sqrt(nsite))
            self.h=HeisenbergH2D(N1,N2,J1=-1.,J1z=1.,J2=-0.25,J2z=0.25,periodic=periodic)
        else:
            raise ValueError()
        #Hilbert Space configuration, # of site, # of spin.
        self.scfg=SpinSpaceConfig([nsite,2])

        #A disguished VMC engine, used for tests.
        self.fv=ExactVMC(self.h)

        #Create a VMC sampling engine.
        self.cgen=RBMConfigGenerator(initial_config=[-1,1]*(nsite/2)+[1]*(nsite%2),nflip=2 if model!='TFI' else 1)
        self.vmc=VMC(self.cgen,nbath=300*nsite,nsample=1000*nsite,nmeasure=nsite,sampling_method='metropolis')

    #@profile
    def test_sr(self,fakevmc=False,debug=False):
        nfeature=2

        #generate a random rbm and the corresponding vector v
        self.rbm=ConvWF(input_shape=(self.nsite,),nfs=[8,32,1024,20], dtype='complex128')

        #get exact ground state vector
        if debug:
            for layer in self.rbm.layers:
                print 'Checking %s'%layer
                diff=check_numdiff(layer, num_check=5, eta=1e-3j+1e-3)
                assert(all(diff))
            print 'Checking Whole Network'
            diff=check_numdiff(self.rbm, num_check=20, eta=1e-4j+1e-4)
            print diff
            #assert(sum(diff)>35)
            pdb.set_trace()
            H=self.fv.get_H()
            e_true,v_true=eigh(H)


        ##################### choose an regulation method ##############################
        #reg_params=('delta',{'lambda0':1e-4})
        #reg_params=('trunc',{'lambda0':0.2,'eps_trunc':1e-3})
        #reg_params=('carleo',{'lambda0':100,'b':0.9})
        reg_params=('sd',{})
        #reg_params=('pinv',{})

        ##################### choose how to compute gradient ##############################
        sr=SR(self.h,self.rbm,num_run=2,handler=self.vmc if not fakevmc else self.fv,reg_params=reg_params)
        
        ##################### choose a optimization method with above gradients known ##############################
        optimizer=RmsProp(wrt=self.rbm.get_variables(),fprime=sr.compute_gradient,step_rate=1e-3,decay=0.9,momentum=0.)
        #optimizer=Adam(wrt=self.rbm.get_variables(),fprime=sr.compute_gradient,step_rate=1e-2)
        #optimizer=GradientDescent(wrt=self.rbm.get_variables(),fprime=sr.compute_gradient,step_rate=1e-3,momentum=0.)

        #optimization variables(b and W) are dumped to an 1darray.
        arr_old=self.rbm.get_variables()
        el=[] # to store error
        for k,info in enumerate(optimizer):
            print 'Running %s-th Iteration.'%k
            #optimizer.step_rate=0.3*0.96**k   #use a decaying learning rate

            #rbm state -> vector state, and then compare it with exact ground state vector
            ei=sr._opq_vals[1]
            if debug:
                err=abs(e_true[0]-ei)/(abs(e_true[0])+abs(ei))
                v=self.rbm.tovec(self.scfg);
                if self.cgen.nflip==2:
                    v=self.fv.project_vec(v)
                v=v/norm(v)
                print 'E/site = %s[%s] (%s), Error = %.4f%%'%(ei/self.nsite,v.conj().dot(H).dot(v)/self.nsite, e_true[0]/self.nsite,err*100)
            else:
                print 'E/site = %s'%(ei/self.nsite)
            el.append(ei)
            #if k>50:optimizer.momentum=0.8   #set momentum for optimizer after some step.

            #see to what extent, variables are changed with respect to those in last iteration.
            arr=self.rbm.get_variables()
            print 'diff rate = %s(norm=%s)'%(norm(arr-arr_old)/norm(arr_old),norm(arr_old))
            arr_old=arr
            if k>800: break
        savetxt('data/err-%s%s.dat'%(self.nsite,'p' if self.periodic else 'o'),el)

if __name__=='__main__':
    random.seed(2)
    t=SRTest(nsite=20, periodic=True,model='J1J2')
    t.test_sr(fakevmc=False, debug=False)
