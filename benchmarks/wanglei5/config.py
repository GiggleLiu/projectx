from models.wanglei5 import WangLei5

num_features_list = [
        [2,2,1],
        [2,2,2],
        [4,2,4],
        [4,4,4],
        [4,4,4,128],
        [4,4,4,128,16],
        [4,4,4,128],
        ]

def modifyconfig_and_getnn(config, bench_id):
    rbm = WangLei5(input_shape=tuple(config['hamiltonian']['size']), num_features=num_features_list[bench_id],
            version='conv', itype='complex128', stride=1,K=4 if bench_id<6 else 8)
    return rbm
