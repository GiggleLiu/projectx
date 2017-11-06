import numpy as np
from models.caizi import CaiZi

def modifyconfig_and_getnn(config, bench_id):
    itype = 'float64'
    num_features1 = [np.prod(config['hamiltonian']['size'])]
    num_features2 = []

    rbm = CaiZi(input_shape=tuple(config['hamiltonian']['size']), itype = itype,
            num_features1=num_features1, num_features2 = num_features2)
    return rbm
