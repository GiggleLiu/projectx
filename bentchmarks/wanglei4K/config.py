from models.wanglei4 import WangLei4

K_list = [1,2,3,4]

def modifyconfig_and_getnn(config, bentch_id):
    rbm = WangLei4(input_shape=tuple(config['hamiltonian']['size']),\
            NF=8, K=K_list[bentch_id], num_features=[32],
            version='conv', dtype='complex128')
    return rbm
