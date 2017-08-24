'''
Problem definition.
'''

import numpy as np
import pdb, time
import scipy.sparse as sps
import matplotlib.pyplot as plt

from poornn.checks import check_numdiff
from qstate.core.utils import unpacknbits_pm, packnbits_pm
from qstate.sampler import SR

def measure_opq(rbm, opq, samples, sign_strategy=('NONE',{})):
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

def analyse_exact(h, do_printsign=False):
    '''using ExactVMC to get exact result.'''
    #get exact ground state vector
    num_eng=10
    nsite=h.nsite
    H = h.get_mat(dense=False).real
    configs, mask = subspace_mask(nsite,mag=h.mag)
    e_true, v_true=sps.linalg.eigsh(H, which='SA', k=num_eng)
    e, v = e_true[0], v_true[:,0]

    if do_printsign:
        print_mat_sign(configs=configs, mat=v[:,np.newaxis]*H*v, signs=sign(v))
    print('Lowest %s energies = %s'%(num_eng, e_true[:num_eng]))
    return H, e, v, configs

def check_rbm_numdiff(rbm, num_check=10):
    '''Check back propagation of RBM.'''
    print('Overall Test')
    diff=check_numdiff(rbm, num_check=num_check, eta=1e-3)
    for layer in rbm.layers:
        print('Testing %s'%layer)
        diff=check_numdiff(layer, num_check=num_check, eta=1e-3)

def plot_wf_distri(h, v0):
    # BUGGY
    v0_ = sort(abs(v0))[::-1]
    plt.plot(v0_**2)
    plt.yscale('log')
    #ylabel(r'$\log(\Psi(x)^2)$')
    plt.ylabel(r'$\Psi(x)^2$')
    plt.ylim(1e-11,1)
    pdb.set_trace()

def compare_wf(vv, v0):
    if vv.dot(v0)<0: vv=-vv

    # pivot
    overlap = v0.dot(vv)
    vv = vv*np.exp(-1j*np.angle(overlap))

    print('|<Psi_0|Psi>|^2 = %s'%abs(overlap)**2)
    plt.title('Wave Function')
    plt.plot(v0, lw=1, color='k')
    plt.plot(vv, lw=1, color='r')
    plt.legend([r'$\Psi_0$', r'$\Psi$'])

def check_sample(rbm, h, samples):
    nsite = h.nsite
    # get rbm wave function
    v=rbm.tovec(mag=h.mag)
    v=v/np.linalg.norm(v)
    H=h.get_mat(dense=False)
    print 'ExactVMC E/site = %s'%(v.conj().T.dot(H.dot(v)).item()/nsite)

    configs = h.configs
    hndim=len(configs)
    wf_sample = np.zeros(hndim)
    wf_sample[h.config_indexer[samples.config_inds]] = np.array(samples.counts, dtype='float64')/samples.num_sample

    plt.plot(abs(v)**2, color ='k')
    plt.plot(wf_sample, color ='r')
    plt.xticks(np.arange(hndim), packnbits_pm(configs))
    plt.legend(['exact', 'vmc'])

def plot_sign_mat(sign_classifier):
    plt.subplot(121)
    plt.title('Raw  Sign')
    sign_classifier.plot_mat_sign(signs = np.ones(v0.shape[0]))
    plt.subplot(122)
    plt.title('True Sign')
    sign_classifier.plot_mat_sign(signs = sign(v0))
    plt.tight_layout()

def plot_count_stat(samples):
    counts = np.array(samples.counts)
    counts = np.sort(counts[counts>0])[::-1]
    plt.plot(counts)

def subspace_mask(nsite, mag):
    configs = unpacknbits_pm(np.arange(2**nsite),nsite)
    mask = configs.sum(axis=1)==mag
    configs = configs[mask]
    return configs, mask

def scatter_vec_phase(v, v0=None, color='r', color0='b'):
    '''
    show the amplitude-phase graph in complex plane.
    '''
    x,y = v.real, v.imag
    plt.scatter(x,y,s=20, color=color)
    if v0 is not None:
        x0,y0 = v0.real, v0.imag
        plt.quiver(x0,y0,x-x0,y-y0, angles='xy', units='xy', scale=1)
        plt.scatter(x0,y0,s=20, color=color0)
    plt.xlabel('$\Re[\Psi]$')
    plt.ylabel('$\Im[\Psi]$')
