from models.wanglei5 import WangLei5
import numpy as np

num_feature_list = [
        [20,20], [20,20], [20, 20], [20, 20],
        ]

nonlinear_list = [
        'sinh','polynomial','hermite', 'hermite',
        ]
momentum_list = [
        0,0,0,0,0,
        ]
usesum_list = [
        False,False,False,False,False,
        ]
poly_order_list = [
        10, 10, 10, 10,
        ]
K_list = [2,2,20,20]

def modifyconfig_and_getnn(config, bench_id):
    rbm = WangLei5(input_shape=tuple(config['hamiltonian']['size']), num_features=num_feature_list[bench_id],
            itype='complex128', stride=1, K=K_list[bench_id],
            usesum=usesum_list[bench_id], nonlinear=nonlinear_list[bench_id],
            momentum=momentum_list[bench_id], eta1=1e-3,
            )
    if bench_id == 3:
        print(rbm.layers[6].factorial_rescale)
        rbm.layers[6].factorial_rescale = True
    return rbm
