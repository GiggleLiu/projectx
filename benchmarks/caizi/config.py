import pdb
import numpy as np
from models.caizi import CaiZi

def modifyconfig_and_getnn(config, bench_id):
    itype = 'float64'
    # set lattice size
    if bench_id==0:
        config['hamiltonian']['size'] = [8]
    if bench_id == 5:
        config['hamiltonian']['size'] = [20]
    if bench_id == 6:
        config['hamiltonian']['size'] = [30]

    version='basic'
    nsite = np.prod(config['hamiltonian']['size'])
    num_features1 = [nsite]
    num_features2 = []

    # set J2
    if bench_id == 2 or bench_id==4:
        config['hamiltonian']['J2'] = 0.8

    # set version
    if bench_id >= 3:
        version='sigmoid'

    rbm = CaiZi(input_shape=tuple(config['hamiltonian']['size']), itype = itype,
            num_features1=num_features1, num_features2 = num_features2, version=version)
    return rbm
