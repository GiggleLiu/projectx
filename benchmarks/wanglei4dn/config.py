from models.wanglei4 import WangLei4

num_features_list = [
        [8, 32],
        [16, 128, 32],
        [16, 128, 64, 32],
        [16, 128, 64, 32, 16],
        [32, 256, 64, 16, 8],
        [1, 16, 256, 16],
        [32, 64, 512, 64, 8],
        [4, 16, 256, 16],
        ]

def modifyconfig_and_getnn(config, bentch_id):
    nfs = num_features_list[bentch_id]
    rbm = WangLei4(input_shape=tuple(config['hamiltonian']['size']),\
            NF=nfs[0], K=4, num_features=nfs[1:],
            version='conv', itype='complex128', dtype0='complex128', dtype1='complex128')
    return rbm
