from models.wanglei5 import WangLei5

num_features_list = [
        [1,128],
        [8],
        [2],
        [8,8],
        [2,2],
        [2,1],
        ]

NP_list = [1,1,1,1,1,1]
NC_list = [0,0,0,1,1,1]

def modifyconfig_and_getnn(config, bench_id):
    rbm = WangLei5(input_shape=tuple(config['hamiltonian']['size']), num_features=num_features_list[bench_id],NP=NP_list[bench_id],NC=NC_list[bench_id],
            version='conv', itype='complex128', stride=1,K=config['hamiltonian']['size'][0], dtype0='complex128', dtype1='complex128', eta0=0.2, eta1=0.2)
    return rbm
