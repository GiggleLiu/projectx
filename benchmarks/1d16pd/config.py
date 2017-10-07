from models.wanglei6 import WangLei6
import numpy as np

num_feature_list = [
        [8], [8, 8], [8, 8], [8, 8], [8, 8, 128],
        [8,8], [8,64,16], [8], [8], [8],
        [8], [8], [8,8,8], [8,64]
        ]
NP_list = [0,0,0,0,0,
        0, 0, 0, 0, 0,
        0, 0, 1, 0,
        ]
NC_list = [1,2,2,2,2,
        2, 1, 1, 1, 1,
        1, 1, 2, 1,
        ]
nonlinear_list_list = [
        ['polynomial','none'],
        ['polynomial','none'],
        ['softplus','none'],
        ['tanh','none'],
        ['softplus','softplus'],

        ['x^3','x^3'],
        ['sinh','arctan','sinh','arctan','sinh'],
        ['sinh','arctan','sinh'],
        ['arctan','sinh','none'],
        ['polynomial','arctan','polynomial'],

        ['arctan','none','exp'],
        ['polynomial_r','arctan','exp'],
        ['none','tan','arctan','log2cosh','exp'],
        ['arctan','tan','log2cosh','exp'],
        ]

def modifyconfig_and_getnn(config, bench_id):
    poly_order = 10
    momentum = np.pi
    itype = 'float64'
    if bench_id==5:
        config['optimize']['optimize_method']='gd'
    if bench_id>10:
        itype = 'complex128'
    if bench_id==13:
        config['optimize']['optimize_method']='rmsprop'

    rbm = WangLei6(input_shape=tuple(config['hamiltonian']['size']), num_features=num_feature_list[bench_id],
            itype=itype,dtype0=itype, dtype1=itype, powerlist=[[1,1],[1,0,1]],
            usesum=False, nonlinear_list=nonlinear_list_list[bench_id], poly_order=poly_order, do_BN=False,
            momentum=momentum, eta0=0.2, eta1=0.2, NP=NP_list[bench_id], NC=NC_list[bench_id])

    if momentum==0: # not ground state
        config['hamiltonian']['EG'] = -6.6889395
    return rbm
