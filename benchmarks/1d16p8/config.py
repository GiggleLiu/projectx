from models.wanglei3 import WangLei3
import numpy as np

powerlist_list = [
        [[1,1],[1,0,1]],
        [[1,1],[1,0,1]],
        [[1,1],[1,0,1]],
        [[1,1],[1,0,1]],
        [[1,1],[1,0,1]],

        [[1,1],[1,0,1]],
        [[1,1],[1,0,1]],
        [[1,1],[1,0,1]],
        [[1,1],[1,0,1]],
        [[1,1],[1,0,1]],

        [[1,1],[1,0,1]],
        [[1,1],[1,0,1]],
        [[1,1],[1,0,1]],
        [[1,1],[1,0,1]],
        [[1,1],[1,0,1]],

        [[1,1],[1,0,1]],
        [[1,1],[1,0,1]],
        ]

num_feature_list = [
        [8], [8], [8], [8], [8],
        [8], [8], [8], [8], [8],
        [8], [8], [8], [8], [8],
        [8], [8],
        ]

nonlinear_list = [
        'sinh','sinh','sinh','sinh','x^3',
        'x^5','sinh','x^5','polynomial','polynomial',
        'polynomial', 'legendre', 'hermite', 'chebyshev', 'laguerre',
        'polynomial', 'hermiteE',
        ]
momentum_list = [
        np.pi, np.pi, np.pi, np.pi, np.pi,
        np.pi, 0, 0, 0, np.pi,
        np.pi, np.pi, np.pi, np.pi, np.pi,
        np.pi, np.pi,
        ]
fixbias_list = [
        True,False,True,False, True,
        True, True, True, True, True,
        False, False, False, False, False,
        False, False,
        ]
usesum_list = [
        True,False,False,True, True,
        True, True, True, True,True,
        False, False, True, False, False,
        False, False,
        ]

def modifyconfig_and_getnn(config, bench_id):
    rbm = WangLei3(input_shape=tuple(config['hamiltonian']['size']), num_features=num_feature_list[bench_id],
            version='conv', itype='complex128', powerlist=powerlist_list[bench_id], stride=1,
            usesum=usesum_list[bench_id], nonlinear=nonlinear_list[bench_id],
            fixbias=fixbias_list[bench_id], momentum=momentum_list[bench_id],eta=0.05 if bench_id==11 else 0.2)

    if momentum_list[bench_id]==0: # not ground state
        config['hamiltonian']['EG'] = -6.6889395
    if bench_id == 15:
        config['optimize']['optimize_method'] = 'rmsprop'
        config['optimize']['step_rate'] = 1e-2
    return rbm
