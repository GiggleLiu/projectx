from models.wanglei3 import WangLei3
import numpy as np

powerlist_list = [
        [[1,1],[1,0,1]],
        [[1,1],[1,0,1]],
        [[1,1],[1,0,1]],
        [[1,1],[1,0,1],[1,0,0,1]],
        ]

num_feature_list = [
        [40], [40], [40], [40],
        ]

nonlinear_list = [
        'sinh','polynomial', 'polynomial', 'legendre',
        ]
momentum_list = [
        0,0,0,0,0,
        ]
fixbias_list = [
        False,False,False,False,False,
        ]
usesum_list = [
        False,False,False,False,False,
        ]

def modifyconfig_and_getnn(config, bench_id):
    rbm = WangLei3(input_shape=tuple(config['hamiltonian']['size']), num_features=num_feature_list[bench_id],
            version='conv', itype='complex128', powerlist=powerlist_list[bench_id], stride=1,
            usesum=usesum_list[bench_id], nonlinear=nonlinear_list[bench_id],
            fixbias=fixbias_list[bench_id], momentum=momentum_list[bench_id], eta=0.01 if nonlinear_list[bench_id]=='sinh' else 0.1)
    if bench_id==2:
        print(rbm.layers[4].factorial_rescale)
        rbm.layers[4].factorial_rescale = True
    return rbm
