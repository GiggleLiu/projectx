'''
Problem definition.
'''

import numpy as np
import pdb, time
import scipy.sparse as sps

from poornn.checks import check_numdiff
from qstate.core.utils import unpacknbits_pm, packnbits_pm
from qstate.sampler import SR
from qstate.sampler.mpiutils import RANK

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

def analyse_exact(h, do_printsign=False, num_eng=1):
    '''using ExactVMC to get exact result.'''
    #get exact ground state vector
    nsite=h.nsite
    H = h.get_mat(dense=False).real
    configs, mask = subspace_mask(nsite,mag=h.mag)
    e_true, v_true=sps.linalg.eigsh(H, which='SA', k=num_eng)
    e, v = e_true[0], v_true[:,0]

    if do_printsign:
        print_mat_sign(configs=configs, mat=v[:,np.newaxis]*H*v, signs=sign(v))
    if RANK==0: print('Lowest %s energies = %s'%(num_eng, e_true[:num_eng]))
    return H, e, v, configs

def check_rbm_numdiff(rbm, num_check=10):
    '''Check back propagation of RBM.'''
    print('Overall Test')
    diff=check_numdiff(rbm, num_check=num_check, eta=1e-3)
    for layer in rbm.layers:
        print('Testing %s'%layer)
        diff=check_numdiff(layer, num_check=num_check, eta=1e-3)

def subspace_mask(nsite, mag):
    configs = unpacknbits_pm(np.arange(2**nsite),nsite)
    mask = configs.sum(axis=1)==mag
    configs = configs[mask]
    return configs, mask

def sign_func_from_vec(configs, v):
    config_inds = packnbits_pm(configs)
    d = dict(zip(config_inds, np.sign(v)))
    return lambda config: d[packnbits_pm(config)]

def space_inversion(config, size, directions=None):
    if directions is None:
        directions = range(len(size))
    config = config.reshape(size)

    for d in directions:
        config = config[(slice(None),)*d+(slice(None,None,-1),)]
    return config.ravel()

def translate(config, size, vec):
    config = config.reshape(size)
    for axis, step in enumerate(vec):
        config = np.roll(config, -step, axis=axis)
    return config.ravel()
