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

def mvar(arr, weights=1, **kwargs):
    '''Weighted Mean and variance.'''
    axis=kwargs.get('axis',0)
    arr = np.asarray(arr)
    weights = np.asarray(weights)[(slice(None),)+(None,)*(arr.ndim-axis-1)]
    N = np.sum(weights)
    mean = np.sum(arr*weights, **kwargs)/N
    var = (abs(arr - mean)**2*weights).sum(**kwargs)/N
    return mean, var

def set_mvar_with_samples(rbm,samples, learning_rate=0.1):
    '''set mean and variance using samples.'''
    if samples is None:
        # initialize mean and variance
        for i,layer in enumerate(rbm.layers):
            if hasattr(layer,'variance'):
                layer.mean = 0.
                layer.variance = 1.
        return
    yss, counts = samples.yss, samples.counts
    for i,layer in enumerate(rbm.layers):
        if not hasattr(layer,'variance'):
            continue
        di = [ys[i] for ys in yss]
        mean, var = mvar(di, weights=counts, axis=0, keepdims=False)
        layer.mean = mean*learning_rate+layer.mean*(1-learning_rate)
        layer.variance = var*learning_rate+layer.variance*(1-learning_rate)

J1J2EG_TABLE = {
        0.0:{20:-8.90438652988, 30:-13.3219630586, 40:-17.7465227719, 100:-44.3229467082},
        0.2:{20:-8.20291625218, 30:-12.2770471706, 40:-16.3566615295, 100:-40.8572924302},
        0.5:{20:-7.5, 30:-11.25, 40:-15, 100:37.5},
        0.8:{20:-8.46127240196, 30:-12.6588455544, 40:-16.8706800952, 100:-42.0700632095},
        }
