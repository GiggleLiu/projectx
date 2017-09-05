import os,pdb,sys
sys.path.insert(0,'../..')

from models.wanglei4 import WangLei4

j2_list = [0.0,0.2,0.5,0.8]

def modifyconfig_and_getnn(config, bentch_id):
    j2 = j2_list[bentch_id]
    config['hamiltonian']['j2'] = j2
    rbm = WangLei4(input_shape=tuple(config['hamiltonian']['size']),\
            NF=8, K=4, num_features=[8],
            version='conv', dtype='complex128')
    return rbm

if __name__=='__main__':
    run()
