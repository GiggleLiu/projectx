from __future__ import division
from models.wanglei4 import WangLei4

nc_list = [1, 2, 4, 8, 16, 32, 64, 24]

def modifyconfig_and_getnn(config, bench_id):
    nfs = [16, 128, 32]
    rbm = WangLei4(input_shape=tuple(config['hamiltonian']['size']),\
            NF=nfs[0], K=4, num_features=nfs[1:],
            version='conv', itype='complex128', dtype0='complex128', dtype1='complex128')

    ncore = nc_list[bench_id]
    config['mpi']['num_core'] = ncore
    config['vmc']['num_vmc_run'] = ncore
    config['vmc']['num_sample'] = 6000//ncore
    return rbm
