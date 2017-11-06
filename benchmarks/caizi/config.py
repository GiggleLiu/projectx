import pdb
import numpy as np
from models.caizi import CaiZi
from utils import J1J2EG_TABLE

def modifyconfig_and_getnn(config, bench_id):
    itype = 'float64'
    # set lattice size
    if bench_id==0:
        config['hamiltonian']['size'] = [8]
    if bench_id == 5:
        config['hamiltonian']['size'] = [20]
    if bench_id == 6:
        config['hamiltonian']['size'] = [30]
    if bench_id == 7:
        config['hamiltonian']['size'] = [40]
    if bench_id == 8:
        config['hamiltonian']['size'] = [40]
        config['hamiltonian']['J2'] = 0.8
    if bench_id == 9:
        use_conv = True

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

    J2 = config['hamiltonian']['J2']
    config['hamiltonian']['EG'] = J1J2EG_TABLE[J2].get(nsite)

    rbm = CaiZi(input_shape=tuple(config['hamiltonian']['size']), itype = itype, use_conv=use_conv,
            num_features1=num_features1, num_features2 = num_features2, version=version)
    return rbm
