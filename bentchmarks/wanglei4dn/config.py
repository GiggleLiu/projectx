from models.wanglei4 import WangLei4

num_features_list = [
        [8, 32],
        [16, 128, 32],
        [16, 128, 64, 32],
        [16, 128, 64, 32, 16],
        [16, 128, 64, 32, 16, 8],
        ]

def modifyconfig_and_getnn(config, bentch_id):
    nfs = num_features_list[bentch_id]
    rbm = WangLei4(input_shape=tuple(config['hamiltonian']['size']),\
            NF=nfs[0], K=4, num_features=nfs[1:],
            version='conv', dtype='complex128')
    return rbm
