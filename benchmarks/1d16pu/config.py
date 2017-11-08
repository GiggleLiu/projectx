import numpy as np
from models.wanglei6 import WangLei6

powerlist_list = [
        [[1,0,1],[1,1]],
        [[1,0,1],[1,1]],
        [[1,0,1],[1,1]],
        [[1,0,1],[1,1]],
        [[1,0,1],[1,1]],

        [[1,0,1],[1,1]],
        ]
num_features_list = [
        [20], [20], [20], [20], [20],
        [20], [20], [20], [20], [20],
        ]
nonlinear_list_list = [
        ['sinh','none','none'],
        ['sinh','none','none'],
        ]

def modifyconfig_and_getnn(config, bench_id):
    is_unitary = False
    momentum = 0
    eta0 = 0.2
    eta1 = 0.2
    NP = 0
    NC = 1
    itype = 'complex128'
    poly_order = 10
    usesum = False
    powerlist = powerlist_list[bench_id]
    num_features = num_features_list[bench_id]
    nonlinear_list = nonlinear_list_list[bench_id]
    soften_gradient = False

    if bench_id == 1:
        momentum=np.pi
    if momentum==0: # not ground state
        config['hamiltonian']['EG'] = -6.6889395

    rbm = WangLei6(input_shape=tuple(config['hamiltonian']['size']), num_features=num_features,
            itype=itype,dtype0=itype, dtype1=itype, powerlist=powerlist,
            usesum=usesum, nonlinear_list=nonlinear_list, poly_order=poly_order, do_BN=False,
            momentum=momentum, eta0=eta0, eta1=eta1, NP=NP, NC=NC,is_unitary=is_unitary,
            soften_gradient = soften_gradient)
    return rbm
