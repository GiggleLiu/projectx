from models.wanglei4 import WangLei4

num_features_list = [
        [16, 32, 128],
        [36, 72, 256],
        [36, 72, 256, 32],
        ]

def modifyconfig_and_getnn(config, bentch_id):
    nfs = num_features_list[bentch_id]
    rbm = WangLei4(input_shape=tuple(config['hamiltonian']['size']),\
            NF=nfs[0], K=6, num_features=nfs[1:],
            version='conv', itype='complex128', dtype0='float64', dtype1='complex128')
    return rbm
