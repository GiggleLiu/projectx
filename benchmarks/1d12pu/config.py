from models.wanglei6 import WangLei6

powerlist_list = [
        [[1]],
        [[1,0,1],[1,1]],
        [[1]],
        [[1,0,1],[1,1]],
        [[1]],

        [[1,0,1],[1,1]],
        [[1]],
        [[1,0,1],[1,1]],
        [[1,0,1],[1,1]],
        [[1,0,1],[1,1]],
        ]
num_features_list = [
        [8], [8], [8], [8], [8],
        [8], [8], [8], [8], [8],
        ]
nonlinear_list_list = [
        ['log2cosh','none','exp'],
        ['log2cosh','none','exp'],
        ['sinh','none','none'],
        ['sinh','none','none'],
        ['log2cosh','none','exp'],

        ['log2cosh','none','exp'],
        ['sinh','none','none'],
        ['sinh','none','none'],
        ['x^3','none','none'],
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
    usesum = True
    powerlist = powerlist_list[bench_id]
    num_features = num_features_list[bench_id]
    nonlinear_list = nonlinear_list_list[bench_id]
    soften_gradient = False

    config['sr']['reg_method'] = 'delta'
    config['optimize']['optimize_method'] = 'gd'
    config['optimize']['step_rate'] = 3e-3

    if bench_id>=4 and bench_id<8:
        soften_gradient = True
    if bench_id>8:
        is_unitary=True

    rbm = WangLei6(input_shape=tuple(config['hamiltonian']['size']), num_features=num_features,
            itype=itype,dtype0=itype, dtype1=itype, powerlist=powerlist,
            usesum=usesum, nonlinear_list=nonlinear_list, poly_order=poly_order, do_BN=False,
            momentum=momentum, eta0=eta0, eta1=eta1, NP=NP, NC=NC,is_unitary=is_unitary,
            soften_gradient = soften_gradient)
    return rbm
