from models.wanglei3 import WangLei3

powerlist_list = [
        [[1]],
        [[1,1]],
        [[1,0,1]],
        [[1,1,1]],
        [[1],[1,1]],
        [[1],[1,1],[1,0,1]],
        [[1],[1,1],[1,0,1],[1,1,1]],
        [[1,1],[1,0,1]],
        [[1,1],[1,0,1]],
        [[1,1],[1,0,1]],
        ]

nonlinear = ['x^3']*8 + ['log2cosh'] + ['polynomial']

def modifyconfig_and_getnn(config, bench_id):
    rbm = WangLei3(input_shape=tuple(config['hamiltonian']['size']), num_features=[4],
            version='conv', itype='complex128', powerlist=powerlist_list[bench_id], stride=1, nonlinear=nonlinear[bench_id],
            with_exp=bench_id==9)
    if bench_id==9:
        print(rbm.layers[4].factorial_rescale)
        rbm.layers[4].factorial_rescale=True
    return rbm
