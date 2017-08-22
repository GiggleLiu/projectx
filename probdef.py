'''
Problem definition.
'''

import numpy as np
import pdb, time
import scipy.sparse as sps
import matplotlib.pyplot as plt

from poornn.checks import check_numdiff
from utils import unpacknbits_pm, packnbits_pm
from sr import SR
from fakevmc import ExactVMC

class ProbDef(object):
    def __init__(self, h, rbm, vmc, sr, optimizer):
        self.h, self.rbm, self.vmc, self.sr, self.optimizer = h, rbm, vmc, sr, optimizer
        self.cache = {}

    def run(self, num_iter = 300,
                plot_mat = False,
                plot_wf = False,
                plot_count_stat=False,
                print_var_diff = False,
                compare_to_exact = False,
                use_exact_sign = False,
                plot_wf_distri = False,
                check_sample = False
            ):

        sr = self.sr

        # plot canvas
        if ((plot_mat or plot_wf or plot_wf_distri or check_sample) and compare_to_exact) or plot_count_stat:
            plt.ion()
            fig=plt.figure(figsize=(10,5))

        # Exact Results
        if compare_to_exact:
            fv = ExactVMC(self.h)
            H, e0, v0, configs = self.analyse_exact(do_printsign=False)
            if plot_wf_distri: self.plot_wf_distri(v0)

        # variables in NN are dumped to an 1darray.
        if print_var_diff: 
            arr_old=self.rbm.get_variables()

        el=[] # to store energy
        print '\nRunning 0-th Iteration.'
        for info in self.optimizer:
            ei = self.cache['opq_vals'][0]
            print 'WA, WB = ',self.rbm.thnn.get_variables()

            if plot_count_stat and not use_exact_sign:
                plt.clf()
                counts = np.array(self.cache['samples'].counts)
                counts = np.sort(counts[counts>0])[::-1]
                plt.plot(counts)
                plt.pause(0.2)

            if plot_mat and compare_to_exact:
                plt.clf()
                plt.subplot(121)
                plt.title('Raw  Sign')
                self.cache['sign_classifier'].plot_mat_sign(signs = np.ones(v0.shape[0]))
                plt.subplot(122)
                plt.title('True Sign')
                self.cache['sign_classifier'].plot_mat_sign(signs = sign(v0))
                plt.tight_layout()
                pdb.set_trace()

            if plot_wf and compare_to_exact:
                self.plot_wf(configs, v0)

            if check_sample:
                self.check_sample(self.cache['samples'])

            if compare_to_exact:
                err=abs(e0-ei)/(abs(e0)+abs(ei))*2
                print('E/site = %s (%s), Error = %.4f%%'%(ei/self.nsite,e0/self.nsite,err*100))
            else:
                print('E/site = %s'%(ei/self.nsite,))
            el.append(ei)

            if print_var_diff:
                #see to what extent, variables are changed with respect to those in last iteration.
                arr=self.rbm.get_variables()
                print 'diff rate = %s(norm=%s)'%(np.linalg.norm(arr-arr_old)/np.linalg.norm(arr_old),np.linalg.norm(arr_old))
                arr_old=arr

            if info['n_iter']>=num_iter:
                break
            print '\nRunning %s-th Iteration.'%(info['n_iter']+1)

        np.savetxt('data/el-%s%s.dat'%(self.nsite,'p' if self.periodic else 'o'),el)
        pdb.set_trace()

    def compute_gradient(self, x):
        '''
        Compute the gradient.
        '''
        vmc, rbm, sr = self.vmc, self.rbm, self.sr

        # update RBM
        rbm.set_variables(x)

        # generate samples probabilities given by this new RBM
        samples = vmc.generate_samples(rbm)

        # perform measurements on operators that needed by stochastic gradient generator.
        opq_vals = self.measure_opq(sr.opq, samples, sign_strategy=('NONE',{}))

        # get gradient from observables
        gradient = sr.get_gradient(opq_vals)

        self.cache['opq_vals'] = opq_vals
        self.cache['samples'] = samples
        return gradient

    def measure_opq(self, opq, samples, sign_strategy=('NONE',{})):
        '''
        Measure Operator Queue.

        Parameters:
            :opq: <OpQueue>,
            :samples: <SampleCollection>
            :sign_strategy: tuple, method_str and configuration parameters.

                * 'NONE', samples themself contain signs, with no parameters.
                * 'QUBO', the sign is decide by solving QUBO problem, with dict parameters 'method', 'num_update', 'num_run' and 'templscales',
                    see qubo_classify function for details.
        '''
        method_str, params = sign_strategy
        rbm = self.rbm

        if method_str == 'QUBO':
            hamiltonian = opq.op_base[0]
            e0 = hailtonian.eval_on_samples(samples, rbm, update_samples=True)
            qubo_config_ = {'method':'sa','num_update':10000, 'num_run':30, 'tempscales':logspace(1.5,-5,200)}
            qubo_config_.update(sign_strategy[1])
            sign_func = qubo_classify(samples, hamiltonian, rbm, qubo_config=qubo_config_)

            # update signs in samples
            samples.signs = sign_func(samples.configs)

        opq_vals = opq.eval_on_samples(samples, rbm, update_samples=True)
        return opq_vals

    def analyse_exact(self, do_printsign=False):
        '''using ExactVMC to get exact result.'''
        from fakevmc import ExactVMC
        fv = ExactVMC(self.h)
        #get exact ground state vector
        num_eng=10
        H0 = fv.get_H(sparse=True).real
        if self.vmc.cgen.nflip == 2:
            mask = fv.subspace_mask()
            H = H0[mask,:][:,mask]
            configs = unpacknbits_pm(np.where(mask)[0][:,np.newaxis],self.nsite)
        else:
            H=H0
            configs = unpacknbits_pm(np.arange(self.nsite)[:,np.newaxis],self.nsite)
        e_true, v_true=sps.linalg.eigsh(H, which='SA', k=num_eng)
        e, v = e_true[0], v_true[:,0]

        if do_printsign:
            print_mat_sign(configs=configs, mat=v[:,np.newaxis]*H*v, signs=sign(v))
        print('Lowest %s energies = %s'%(num_eng, e_true[:num_eng]))
        return H0, e, v, configs

if __name__=='__main__':
    t=SRTest(nsite=6,periodic=True,model='J1J2')
    t.run()
