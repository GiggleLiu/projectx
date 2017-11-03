from models.wanglei6 import WangLei6
import numpy as np

num_feature_list = [
        [8], [8, 8], [8, 8], [8, 8], [8, 8, 128],
        [8,8], [8,64,16], [8], [8], [8],
        [8], [8], [8,8,8], [8,64], [8,64],
        [8,64], [8,8], [8,64], [8,8], [8,8],
        [8,8], [8,8], [8,8], [8,8], [8,8],
        [8,8], [8, 8], [8, 128, 64, 32, 16, 8], [8,64], [8, 8],
        [8,8], [8,8], [8,8], [8,8], [8,8],
        [8,8], [8], [8,64,8], [8,8], [8,8,8],
        [8,8],
        ]
NP_list = [0,0,0,0,0,
        0, 0, 0, 0, 0,
        0, 0, 1, 0, 0,
        0, 1, 0, 1, 1,
        1, 1, 1, 1, 1,
        1, 0, 0, 0, 1,
        1, 1, 1, 1, 1,
        1, 0, 0, 1, 1,
        1,
        ]
NC_list = [1,2,2,2,2,
        2, 1, 1, 1, 1,
        1, 1, 2, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        1,
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
        ['sinh','none','relu','none'],

        ['sinh','none','none','none'],
        ['relu','polynomial','relu','none'],
        ['polynomial_r','relu','relu','none'],
        ['relu','polynomial_r','relu','none'],
        ['relu','polynomial_r','none','none'],

        ['none','polynomial_r','none','none'],
        ['none','ks_logcosh','none','polynomial'],
        ['none','ks_logcosh','polynomial','none'],
        ['none','ks_logcosh','exp','none'],
        ['none','ks_logcosh','none','exp'],

        ['none','logcosh','none','exp'],
        ['IsingRG2D','IsingRG2D','IsingRG2D','IsingRG2D'],
        ['IsingRG2D']*8,
        ['ks_IsingRG2D']*4,
        ['ks_x^1/3','x^3','ks_x^1/3','x^3'],

        ['x^3','none','ks_x^1/3','x^3'],
        ['none','ks_IsingRG2D','ks_x^1/3','x^3'],
        ['none','XLogcosh','XLogcosh','XLogcosh'],
        ['none','XLogcosh','ks_IsingRG2D','sinh'],
        ['none','XLogcosh','relu','exp'],

        ['real','polynomial','arctan','sinh'],
        ['XLogcosh','XLogcosh','exp'],
        ['IsingRG2D']*4+['x^3'],
        ['IsingRG2D']*4+['polynomial'],
        ['IsingRG2D']*6,

        ['none','polynomial_r','none','none'],
        ]

def modifyconfig_and_getnn(config, bench_id):
    poly_order = 10
    momentum = np.pi
    powerlist = [[1,1],[1,0,1]]
    itype = 'float64'
    eta0 = 0.2
    eta1 = 0.2
    is_unitary=False
    if bench_id==5:
        config['optimize']['optimize_method']='gd'
    if bench_id>10:
        itype = 'complex128'
    if bench_id==13:
        config['optimize']['optimize_method']='rmsprop'
    if (bench_id>=18 and bench_id<=25) or bench_id>=29:
        if bench_id!=36:
            powerlist=None
    if bench_id>=23:
        if bench_id<=33:
            config['optimize']['step_rate']=1e-3
        config['optimize']['optimize_method']='rmsprop'
        #if bench_id==35:
        #    config['sr']['reg_method']='delta'
        #    config['optimize']['step_rate']=1e-2
        #    config['optimize']['optimize_method']='gd'
        if bench_id==36:
            config['optimize']['step_rate']=3e-3
    if bench_id>25:
        eta1 = 1e-1
        if bench_id==30:
            eta1=0.03
    if bench_id>=39:
        is_unitary=True

    rbm = WangLei6(input_shape=tuple(config['hamiltonian']['size']), num_features=num_feature_list[bench_id],
            itype=itype,dtype0=itype, dtype1=itype, powerlist=powerlist,
            usesum=False, nonlinear_list=nonlinear_list_list[bench_id], poly_order=poly_order, do_BN=False,
            momentum=momentum, eta0=eta0, eta1=eta1, NP=NP_list[bench_id], NC=NC_list[bench_id],is_unitary=is_unitary)

    if momentum==0: # not ground state
        config['hamiltonian']['EG'] = -6.6889395
    return rbm
