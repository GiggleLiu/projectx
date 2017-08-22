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
from poorrbm import *
from models import *
from toymodel import ExactVMC
from sr import *
from cgen import *
from sampler.vmc import *
from group import TIGroup,NoGroup
from climin import RmsProp,GradientDescent,Adam
from poornn.checks import check_numdiff
from signlib import qubo_classify, map_classifier, check_classifier, print_mat_sign
from probdef import ProbDef

from test_vmc import analyse_sampling

random.seed(2)

class SRTest(ProbDef):
    def __init__(self,nsite,periodic,model='AFH'):
        #Generate a model
        self.nsite,self.periodic=nsite,periodic
        self.model=model
        if model=='AFH':
            self.h=HeisenbergH(nsite=nsite,J1=1.,J1z=1.,periodic=periodic)
        elif model=='J1J2':
            self.h=HeisenbergH(nsite=nsite,J1=1.,J1z=1.,J2=0.5,J2z=0.5,periodic=periodic)
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
        self.vmc=VMC(cgen,nbath=100*nsite,nsample=1000*nsite,nmeasure=nsite,sampling_method='metropolis')

    def batch_vector_solve(self, do_printsign=False, devide_sign=False):
        num_iter = 20000
        percent = 0.4
        #get exact ground state vector
        H = self.fv.get_H(sparse=True).real
        if self.vmc.cgen.nflip == 2:
            mask = self.fv.subspace_mask()
            H = H[mask,:][:,mask]
            configs = unpacknbits_pm(where(mask)[0][:,newaxis],self.nsite)
        else:
            configs = unpacknbits_pm(arange(self.nsite)[:,newaxis],self.nsite)
        e_true, v_true=sps.linalg.eigsh(H, which='SA', k=1)
        et, vt = e_true[0], v_true[:,0]
        num_batch=int(len(vt)*percent)

        v0=random.random(len(vt))
        v0=v0/linalg.norm(v0)

        def compute_gradient(v):
            # select a batch
            #indices = argsort(v)[:-num_batch]
            indices = where(random.random(len(v0))<percent)[0]
            Hc=H[indices,:][:,indices]
            #v = sps.linalg.eigsh(Hc, which='SA', k=1)
            g=zeros_like(v)
            if not devide_sign:
                vc = v[indices]
                Hv = Hc.dot(vc)
                E = vc.conj().dot(Hv)
                g[indices] = (Hv-E*vc)/linalg.norm(vc)**2
            else:
                pdb.set_trace()
            if do_printsign:
                print_mat_sign(configs=unpacknbits_pm(indices), mat=abs(vc[:,newaxis])*H*abs(vc), signs=sign(vc))
            return g

        ##################### choose a optimization method with above gradients known ##############################
        #optimizer=RmsProp(wrt=v0,fprime=compute_gradient,step_rate=1e-2,decay=0.9)
        #optimizer=Adam(wrt=v0,fprime=compute_gradient,step_rate=1e-3)
        #optimizer=GradientDescent(wrt=v0,fprime=compute_gradient,step_rate=1e-3,momentum=0.9)

        for i in optimizer:
            v0/=linalg.norm(v0)
            print('%s, Energy = %.4f (%.4f), overlap = %.4f'%(i['n_iter'],v0.conj().dot(H.dot(v0)), et,(vt.conj().dot(v0))**2))
            if i['n_iter']>num_iter: break

        return v0.conj().dot(H.dot(v0)), v0

    def analyse_exact(self, do_printsign=True):
        '''using ExactVMC to get exact result.'''
        #get exact ground state vector
        num_eng=10
        H = self.fv.get_H(sparse=True).real
        if self.vmc.cgen.nflip == 2:
            mask = self.fv.subspace_mask()
            H = H[mask,:][:,mask]
            configs = unpacknbits_pm(where(mask)[0][:,newaxis],self.nsite)
        else:
            configs = unpacknbits_pm(arange(self.nsite)[:,newaxis],self.nsite)
        e_true, v_true=sps.linalg.eigsh(H, which='SA', k=num_eng)
        e, v = e_true[0], v_true[:,0]

        if do_printsign:
            print_mat_sign(configs=configs, mat=v[:,newaxis]*H*v, signs=sign(v))
        exact_classifier=map_classifier(configs, sign(v))
        print('Lowest %s energies = %s'%(num_eng, e_true[:num_eng]))
        return e, v, configs, exact_classifier

    def scale_wf(self, sites):
        tol=1e-10
        cut_points = []
        for nsite in sites:
            self.h.nsite=nsite
            self.fv = ExactVMC(self.h)
            e0, v0, configs, exact_classifier = self.analyse_exact(do_printsign=False)
            v0_ = sort(abs(v0))[::-1]
            cut_point = sum(v0_**2>tol)
            cut_points.append(cut_point)
        ion()
        fig=figure(figsize=(10,5))
        plot(sites, cut_points)
        ylabel(r'N')
        yscale('log')
        pdb.set_trace()
        savefig('data/SCALE_J2%s.pdf'%(self.h.J2))

    def test_sr(self,fakevmc=False):
        nfeature = 4
        do_check_numdiff=False
        plot_mat = False
        plot_wf = True
        plot_count_stat=False
        print_var_diff = False
        compare_to_exact = True
        use_exact_sign = True
        plot_wf_distri = True

        # plot
        if ((plot_mat or plot_wf or plot_wf_distri) and compare_to_exact) or plot_sign_stat:
            ion()
            fig=figure(figsize=(10,5))

        # Generate a random rbm and the corresponding vector v
        self.rbm=RBM(input_shape=(self.nsite,),num_feature_hidden=nfeature, dtype='float64')

        if do_check_numdiff:
            diff=check_numdiff(self.rbm, num_check=500, eta=1e-3)
            for layer in self.rbm.layers:
                print 'Testing %s'%layer
                diff=check_numdiff(layer, num_check=10, eta=1e-3)

        # Exact Results
        if compare_to_exact:
            e0, v0, configs, exact_classifier = self.analyse_exact(do_printsign=self.nsite<=6)
            if plot_wf_distri:
                v0_ = sort(abs(v0))[::-1]
                clf()
                plot(v0_**2)
                yscale('log')
                #ylabel(r'$\log(\Psi(x)^2)$')
                ylabel(r'$\Psi(x)^2$')
                ylim(1e-11,1)
                pdb.set_trace()
                savefig('data/AMP_J2%s_N%s.pdf'%(self.h.J2,self.nsite))

        ##################### choose an regulation method ##############################
        reg_params=('delta',{'lambda0':1e-4})
        #reg_params=('trunc',{'lambda0':0.2,'eps_trunc':1e-3})
        #reg_params=('carleo',{'lambda0':100,'b':0.9})
        #reg_params=('sd',{})
        #reg_params=('pinv',{})

        ##################### choose how to compute gradient ##############################
        #sr=SR(self.h,self.rbm,handler=self.vmc if not fakevmc else self.fv,reg_params=reg_params, train_sign=True)
        sr=SR(self.h,self.rbm,handler=self.vmc if not fakevmc else self.fv,reg_params=reg_params, train_sign=exact_classifier if use_exact_sign else True)
        self.sr=sr
        
        ##################### choose a optimization method with above gradients known ##############################
        #optimizer=RmsProp(wrt=self.rbm.get_variables(),fprime=sr.compute_gradient,step_rate=1e-3,decay=0.9,momentum=0.)
        #optimizer=Adam(wrt=self.rbm.get_variables(),fprime=sr.compute_gradient,step_rate=1e-2)
        optimizer=GradientDescent(wrt=self.rbm.get_variables(),fprime=sr.compute_gradient,step_rate=1e-2,momentum=0.)

        #optimization variables(b and W) are dumped to an 1darray.
        if print_var_diff: 
            arr_old=self.rbm.get_variables()
        el=[] # to store error

        print '\nRunning 0-th Iteration.'
        for k,info in enumerate(optimizer):
            #rbm state -> vector state, and then compare it with exact ground state vector
            #v=self.rbm.tovec(self.scfg); v=v/norm(v)
            compare_to_exact = k%10 == 0
            plot_count_stat = k%10 == 0
            ei=sr._opq_vals[1]

            if plot_count_stat and not use_exact_sign:
                clf()
                counts = array(sr._sign_classifier.samples.counts)
                counts = sort(counts[counts>0])[::-1]
                plot(counts)
                pause(1)

            #generate samples and test correctness
            if check_classifier and compare_to_exact and not use_exact_sign:
                sr._sign_classifier.self_check(configs, sign(v0))

            if plot_mat and compare_to_exact:
                clf()
                subplot(121)
                title('Raw  Sign')
                sr._sign_classifier.plot_mat_sign(signs = ones(v0.shape[0]))
                subplot(122)
                title('True Sign')
                sr._sign_classifier.plot_mat_sign(signs = sign(v0))
                tight_layout()
                pdb.set_trace()

            if plot_wf and compare_to_exact:
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
                pause(1)
            #lambda config: y[mapping[packnbits_pm(config, self.nsite).item()]]

            if compare_to_exact:
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

            if k>300:
                if plot_wf: savefig('data/WF_J2%s_N%s.pdf'%(self.h.J2,self.nsite))
                if plot_mat: savefig('data/MAT_J2%s_N%s.pdf'%(self.h.J2,self.nsite))
                break
            print '\nRunning %s-th Iteration.'%(k+1)

        savetxt('data/el-%s%s.dat'%(self.nsite,'p' if self.periodic else 'o'),el)
        pdb.set_trace()

if __name__=='__main__':
    t=SRTest(nsite=6,periodic=True,model='J1J2')
    #t.test_sr()
    t.scale_wf(arange(8,22,2))
    #t.batch_vector_solve(do_printsign=False, devide_sign=False)
