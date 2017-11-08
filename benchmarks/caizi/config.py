import pdb
import numpy as np
from models.caizi import CaiZi
from utils import J1J2EG_TABLE

def modifyconfig_and_getnn(config, bench_id):
    itype = 'float64'
    use_conv=[False,False]
    preprocessing = False
    # set lattice size
    if bench_id==0:
        config['hamiltonian']['size'] = [8]
    if bench_id == 5:
        config['hamiltonian']['size'] = [20]
    if bench_id == 6:
        config['hamiltonian']['size'] = [30]
        config['vmc']['num_sample'] = 800
        config['vmc']['num_bath_sweep'] = 200
    if bench_id == 7:
        config['hamiltonian']['size'] = [40]
    if bench_id == 8:
        config['hamiltonian']['size'] = [40]
        config['hamiltonian']['J2'] = 0.8
    if bench_id == 9:
        use_conv = [True, False]
    if bench_id == 10:
        use_conv = [True, True]
    if bench_id == 11:
        config['hamiltonian']['size'] = [40]
        use_conv = [True, True]
    if bench_id == 12:
        config['hamiltonian']['size'] = [40]
        config['hamiltonian']['J2'] = 0.8
        use_conv = [True, True]
    if bench_id == 13:    
        use_conv = [True, True]
        preprocessing = True
    if bench_id == 14:    
        config['hamiltonian']['size'] = [40]
        config['hamiltonian']['J2'] = 0.8
        use_conv = [True, True]
        preprocessing = True
    if bench_id == 15:    
        use_conv = [True, True]
        preprocessing = True
        config['hamiltonian']['size'] = [40]
    if bench_id == 16:
        config['hamiltonian']['size'] = [30]
        config['hamiltonian']['J2'] = 0.8
        preprocessing = True
        config['optimize']['step_rate']*=3
    if bench_id == 17:
        use_conv = [True, True]
        config['hamiltonian']['size'] = [40]
        config['hamiltonian']['J2'] = 0.2
        preprocessing = True

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
    if use_conv[0]:
        num_features1 = [10]

    J2 = config['hamiltonian']['J2']
    config['hamiltonian']['EG'] = J1J2EG_TABLE[J2].get(nsite)

    rbm = CaiZi(input_shape=tuple(config['hamiltonian']['size']), itype = itype, use_conv=use_conv,
            num_features1=num_features1, num_features2 = num_features2, version=version, preprocessing=preprocessing)
    return rbm
