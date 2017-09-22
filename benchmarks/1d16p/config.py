from models.wanglei3 import WangLei3
import numpy as np

powerlist_list = [
        [[1,1],[1,0,1]],
        [[1,1],[1,0,1]],
        [[1,1],[1,0,1]],
        [[1,1],[1,0,1],[1]],
        [[1,1],[1,0,1]],
        ]

num_feature_list = [[4,4],[4,64],[8],[8,64],[8]]

def modifyconfig_and_getnn(config, bench_id):
    rbm = WangLei3(input_shape=tuple(config['hamiltonian']['size']), num_features=num_feature_list[bench_id],
            version='conv', itype='complex128', powerlist=powerlist_list[bench_id], stride=1,
            usesum=False, nonlinear='x^5',
            fixbias=False, momentum=0)
    return rbm
